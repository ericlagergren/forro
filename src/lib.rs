//! The Forró cipher.
//!
//! [Forró] is an add-rotate-xor (ARX) cipher recently introduced
//! by Murilo et al. at Asiacrypt 2022. It is similar to the
//! ChaCha cipher, but offers better diffusion and requires
//! fewer rounds. In general, Forró saves about two rounds for
//! every seven ChaCha rounds. In other words, Forró14 is
//! equivalent to ChaCha20, Forró10 is equivalent to ChaCha12,
//! and so on.
//!
//! This crate implements the non-authenticated stream ciphers
//! and the AEAD APIs.
//!
//! # Warning
//!
//! Forró is a very new cipher and has not had much independent
//! cryptanalysis. This library is also undertested (e.g., it
//! does not have negative tests.) You should not use this in
//! production.
//!
//! [Forró]: https://link.springer.com/article/10.1007/s00145-023-09455-5

#![cfg_attr(docs, feature(doc_cfg))]
#![cfg_attr(feature = "error_in_core", feature(error_in_core))]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![deny(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::implicit_saturating_sub,
    clippy::panic,
    clippy::unwrap_used,
    missing_docs,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]
#![forbid(unsafe_code)]

use {
    byteorder::{ByteOrder, LittleEndian},
    cfg_if::cfg_if,
    core::{
        cmp, fmt,
        iter::{zip, Iterator},
        mem,
        ops::{Deref, DerefMut},
        result::Result,
    },
    poly1305::{
        universal_hash::{self, KeyInit, UniversalHash},
        Poly1305,
    },
};

cfg_if! {
    if #[cfg(feature = "error_in_core")] {
        use core::error;
    } else if #[cfg(feature = "std")] {
        use std::error;
    }
}

/// Like [`assert!`], but forces a compile-time error.
macro_rules! const_assert {
    ($($tt:tt)*) => {
        const _: () = assert!($($tt)*);
    }
}
// In order to support, e.g., 16-bit CPUs we'll need to rethink
// how we compute some of the constants.
const_assert!(mem::size_of::<usize>() >= 4);

/// An error returned by this crate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// The plaintext is too large.
    PlaintextTooLarge,
    /// The ciphertext is too large.
    CiphertextTooLarge,
    /// The additional data is too large.
    AdditionalDataTooLarge,
    /// The output buffer is too small.
    BufferTooSmall,
    /// The message could not be authenticated.
    Authentication,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlaintextTooLarge => write!(f, "plaintext too large"),
            Self::CiphertextTooLarge => write!(f, "ciphertext too large"),
            Self::AdditionalDataTooLarge => {
                write!(f, "additional data too large")
            }
            Self::BufferTooSmall => write!(f, "output buffer too small"),
            Self::Authentication => write!(f, "message authentication failure"),
        }
    }
}

#[cfg_attr(docs, doc(cfg(any(feature = "error_in_core", feature = "std"))))]
#[cfg(any(feature = "error_in_core", feature = "std"))]
impl error::Error for Error {}

impl From<universal_hash::Error> for Error {
    fn from(_err: universal_hash::Error) -> Self {
        Self::Authentication
    }
}

/// The size in octets of a Forró block.
pub const BLOCK_SIZE: usize = 64;

/// The size in octets of a key.
pub const KEY_SIZE: usize = 32;

/// Returns the number of blocks in `len`, saturated to
/// [`u32::MAX`].
const fn num_blocks(len: usize) -> u32 {
    let mut n = len / BLOCK_SIZE;
    if len % BLOCK_SIZE != 0 {
        n += 1;
    }
    if n > (u32::MAX as usize) {
        u32::MAX
    } else {
        n as u32
    }
}

/// The Forró stream cipher using 14 rounds.
pub type Forro14 = Forro<14>;

/// The Forró stream cipher reduced to 10 rounds.
pub type Forro10 = Forro<10>;

/// The Forró stream cipher.
#[cfg_attr(feature = "zeroize", derive(zeroize::ZeroizeOnDrop))]
pub struct Forro<const R: usize> {
    key: [u32; 8],
    t: [u32; 2],
    v: [u32; 2],
    /// Contains leftover keystream.
    tmp: [u8; BLOCK_SIZE],
    /// Number of bytes available in `tmp`.
    len: usize,
    /// Block counter.
    blocks: u32,
    /// 64-bit counter?
    count64: bool,
}

impl<const R: usize> Forro<R> {
    /// The size in octets of a nonce.
    pub const NONCE_SIZE: usize = 12;

    /// The maximum allowed number of plaintext blocks.
    pub const P_BLOCKS_MAX: u32 = u32::MAX;

    /// The size in octets of the largest allowed plaintext.
    pub const P_MAX: u64 = (1 << 38) - 64;

    /// Creates a new stream cipher.
    #[inline]
    pub fn new(key: &[u8; 32], nonce: &[u8; 12]) -> Self {
        Self::new_with_ctr(key, nonce, 0)
    }

    fn new_with_ctr(key: &[u8; 32], nonce: &[u8; 12], ctr: u32) -> Self {
        Self {
            key: key_to_words(key),
            t: [ctr, LittleEndian::read_u32(&nonce[0..4])],
            v: [
                LittleEndian::read_u32(&nonce[4..8]),
                LittleEndian::read_u32(&nonce[8..12]),
            ],
            tmp: [0u8; BLOCK_SIZE],
            len: 0,
            blocks: 0,
            count64: false,
        }
    }

    #[cfg(test)]
    fn new_with_64bit_nonce(key: &[u8; 32], nonce: &[u8; 8], ctr: u64) -> Self {
        Self {
            key: key_to_words(key),
            t: [ctr as u32, (ctr >> 32) as u32],
            v: [
                LittleEndian::read_u32(&nonce[0..4]),
                LittleEndian::read_u32(&nonce[4..8]),
            ],
            tmp: [0u8; BLOCK_SIZE],
            len: 0,
            blocks: 0,
            count64: true,
        }
    }

    /// XORs each byte in `src` with a byte from the keystream.
    ///
    /// `dst` must be at least as long as `src`.
    ///
    /// Multiple calls to [`xor`][Self::xor] behave as if the
    /// concatenation of multiple `src` buffers were passed in
    /// single run.
    ///
    /// The ciphertext is NOT authenticated. For an AEAD, see one
    /// of the [`ForroPoly1305`] variants.
    pub fn xor(
        &mut self,
        mut dst: &mut [u8],
        mut src: &[u8],
    ) -> Result<(), Error> {
        if dst.len() < src.len() {
            return Err(Error::BufferTooSmall);
        }
        dst = &mut dst[..src.len()];

        // Is there any carryover?
        if self.len != 0 {
            let n = cmp::min(self.len, src.len());
            xor(dst, src, &self.tmp[self.tmp.len() - self.len..]);
            self.len -= n;
            dst = &mut dst[n..];
            src = &src[n..];
        }
        if src.is_empty() {
            return Ok(());
        }

        let blocks = num_blocks(src.len());
        self.blocks += match self.blocks.checked_add(blocks) {
            Some(n) => n,
            None => return Err(Error::PlaintextTooLarge),
        };

        let mut ctx = State::new(&self.key, &self.t, &self.v);

        // Full blocks.
        let mut dst = dst.chunks_exact_mut(BLOCK_SIZE);
        let mut src = src.chunks_exact(BLOCK_SIZE);
        for (dst, src) in zip(&mut dst, &mut src) {
            xor(dst, src, &ctx.core::<R>());
            ctx.incr_ctr(self.count64);
        }

        // Partial block.
        let src = src.remainder();
        if !src.is_empty() {
            let dst = dst.into_remainder();

            self.tmp[..src.len()].copy_from_slice(src);
            xor_in_place(&mut self.tmp, &ctx.core::<R>());
            dst.copy_from_slice(&self.tmp[..src.len()]);
            self.len = self.tmp.len() - src.len();
        }

        Ok(())
    }

    /// Same as [`xor`][Self::xor], but performed in-place.
    pub fn xor_in_place(&mut self, mut data: &mut [u8]) -> Result<(), Error> {
        // Is there any carryover?
        if self.len != 0 {
            let n = cmp::min(self.len, data.len());
            xor_in_place(data, &self.tmp[self.tmp.len() - self.len..]);
            self.len -= n;
            data = &mut data[n..];
        }
        if data.is_empty() {
            return Ok(());
        }

        let blocks = num_blocks(data.len());
        self.blocks += match self.blocks.checked_add(blocks) {
            Some(n) => n,
            None => return Err(Error::PlaintextTooLarge),
        };

        let mut ctx = State::new(&self.key, &self.t, &self.v);

        // Full blocks.
        let mut data = data.chunks_exact_mut(BLOCK_SIZE);
        for chunk in &mut data {
            xor_in_place(chunk, &ctx.core::<R>());
            ctx.incr_ctr(self.count64);
        }

        // Partial block.
        let data = data.into_remainder();
        if !data.is_empty() {
            self.tmp[..data.len()].copy_from_slice(data);
            xor_in_place(&mut self.tmp, &ctx.core::<R>());
            data.copy_from_slice(&self.tmp[..data.len()]);
            self.len = self.tmp.len() - data.len();
        }

        Ok(())
    }

    fn hash(data: &[u8; 16], key: &[u8; 32]) -> [u8; 32] {
        let mut c = [0u32; 4];
        for (w, chunk) in zip(c.iter_mut(), data.chunks_exact(4)) {
            *w = LittleEndian::read_u32(chunk);
        }
        let mut ctx =
            State::new(&key_to_words(key), &[c[0], c[1]], &[c[2], c[3]]);
        ctx.rounds::<R>();
        let mut out = [0u8; 32];
        LittleEndian::write_u32(&mut out[0..4], ctx[0]);
        LittleEndian::write_u32(&mut out[4..8], ctx[1]);
        LittleEndian::write_u32(&mut out[8..12], ctx[2]);
        LittleEndian::write_u32(&mut out[12..16], ctx[3]);
        LittleEndian::write_u32(&mut out[16..20], ctx[12]);
        LittleEndian::write_u32(&mut out[20..24], ctx[13]);
        LittleEndian::write_u32(&mut out[24..28], ctx[14]);
        LittleEndian::write_u32(&mut out[28..32], ctx[15]);
        out
    }
}

/// The Forró stream cipher using 14 rounds with an extended
/// nonce.
pub type XForro14 = XForro<14>;

/// The Forró stream cipher reduced to 10 rounds with an extended
/// nonce.
pub type XForro10 = XForro<10>;

/// The Forró stream cipher with an extended nonce.
#[cfg_attr(feature = "zeroize", derive(zeroize::ZeroizeOnDrop))]
pub struct XForro<const R: usize>(Forro<R>);

impl<const R: usize> XForro<R> {
    /// The size in octets of a nonce.
    pub const NONCE_SIZE: usize = 24;

    /// The maximum allowed number of plaintext blocks.
    pub const P_BLOCKS_MAX: u32 = u32::MAX;

    /// The size in octets of the largest allowed plaintext.
    pub const P_MAX: u64 = (1 << 38) - 64;

    /// Creates a new stream cipher.
    #[inline]
    pub fn new(key: &[u8; 32], nonce: &[u8; 24]) -> Self {
        let (key, nonce) = extend::<R>(key, nonce);
        Self(Forro::new(&key, &nonce))
    }

    /// XORs each byte in `src` with a byte from the keystream.
    ///
    /// `dst` must be at least as long as `src`.
    ///
    /// Multiple calls to [`xor`][Self::xor] behave as if the
    /// concatenation of multiple `src` buffers were passed in
    /// single run.
    ///
    /// The ciphertext is NOT authenticated. For an AEAD, see one
    /// of the [`XForroPoly1305`] variants.
    #[inline]
    pub fn xor(&mut self, dst: &mut [u8], src: &[u8]) -> Result<(), Error> {
        self.0.xor(dst, src)
    }

    /// Same as [`xor`][Self::xor], but performed in-place.
    #[inline]
    pub fn xor_in_place(&mut self, data: &mut [u8]) -> Result<(), Error> {
        self.0.xor_in_place(data)
    }
}

/// The Forró AEAD using 14 rounds.
pub type Forro14Poly1305 = ForroPoly1305<14>;

/// The Forró AEAD using 10 rounds.
pub type Forro10Poly1305 = ForroPoly1305<10>;

/// The Forró AEAD.
#[cfg_attr(feature = "zeroize", derive(zeroize::ZeroizeOnDrop))]
pub struct ForroPoly1305<const R: usize>([u8; 32]);

impl<const R: usize> ForroPoly1305<R> {
    /// The size in octets of a nonce.
    pub const NONCE_SIZE: usize = 12;

    /// The size in octets of an authentication tag.
    pub const TAG_SIZE: usize = 16;

    /// The maximum size in octets of a plaintext.
    pub const P_MAX: u64 = (1 << 38) - 64;

    /// The maximum size in octets of a ciphertext.
    pub const C_MAX: u64 = Self::P_MAX + Self::TAG_SIZE as u64;

    /// The maximum size in octets of additional data.
    pub const A_MAX: u64 = u64::MAX;

    /// Creates an instance of the AEAD.
    #[inline]
    pub fn new(key: &[u8; 32]) -> Self {
        Self(*key)
    }

    /// Encrypts and authenticates `plaintext`, writing the
    /// result to `dst`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least [`TAG_SIZE`][Self::TAG_SIZE]
    ///   octets longer than `plaintext`.
    /// - `plaintext` must be at most [`P_MAX`][Self::P_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn seal(
        &self,
        dst: &mut [u8],
        nonce: &[u8; 12],
        plaintext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        if plaintext.len() as u64 > Self::P_MAX {
            return Err(Error::PlaintextTooLarge);
        }
        if additional_data.len() as u64 > Self::A_MAX {
            return Err(Error::AdditionalDataTooLarge);
        }
        // This will not overflow since `plaintext.len()` is less
        // than `P_MAX` and `P_MAX` + `TAG_SIZE` will not
        // overflow.
        if dst.len() < plaintext.len() + Self::TAG_SIZE {
            return Err(Error::BufferTooSmall);
        }

        let (dst, tag) = dst.split_at_mut(dst.len() - Self::TAG_SIZE);
        self.seal_scatter(
            dst,
            tag.try_into().expect("bug: incorrect tag size"),
            nonce,
            plaintext,
            additional_data,
        )
    }

    /// Decrypts and authenticates `ciphertext`, writing the
    /// result to `dst`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least `ciphertext.len()` -
    ///   [`TAG_SIZE`][Self::TAG_SIZE] bytes long.
    /// - `ciphertext` must be at most [`C_MAX`][Self::C_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn open(
        &self,
        dst: &mut [u8],
        nonce: &[u8; 12],
        ciphertext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        if ciphertext.len() < Self::TAG_SIZE {
            // Obviously cannot be authenticated if there isn't
            // a tag.
            return Err(Error::Authentication);
        }
        if ciphertext.len() as u64 > Self::C_MAX {
            return Err(Error::CiphertextTooLarge);
        }
        if additional_data.len() as u64 > Self::A_MAX {
            return Err(Error::AdditionalDataTooLarge);
        }
        // Cannot overflow since ciphertext is at least
        // `TAG_SIZE` bytes long.
        if dst.len() < ciphertext.len() - Self::TAG_SIZE {
            return Err(Error::BufferTooSmall);
        }

        let (ciphertext, tag) =
            ciphertext.split_at(ciphertext.len() - Self::TAG_SIZE);
        self.open_gather(
            dst,
            tag.try_into().expect("bug: incorrect tag size"),
            nonce,
            ciphertext,
            additional_data,
        )
    }

    /// Encrypts and authenticates `plaintext`.
    ///
    /// The result (less the authentication tag) is written to
    /// `dst` and the authentication tag is written to `tag`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least as long as `plaintext`.
    /// - `plaintext` must be at most [`P_MAX`][Self::P_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    pub fn seal_scatter(
        &self,
        dst: &mut [u8],
        tag: &mut [u8; 16],
        nonce: &[u8; 12],
        plaintext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        if dst.len() < plaintext.len() {
            return Err(Error::BufferTooSmall);
        }
        if plaintext.len() as u64 > Self::P_MAX {
            return Err(Error::PlaintextTooLarge);
        }
        if additional_data.len() as u64 > Self::A_MAX {
            return Err(Error::AdditionalDataTooLarge);
        }

        let mut s = Forro::<R>::new(&self.0, nonce);

        let mut mac = {
            let mut poly_key = [0u8; 32];
            s.xor_in_place(&mut poly_key)?;
            Poly1305::new((&poly_key).into())
        };
        s.xor(&mut dst[..plaintext.len()], plaintext)?;

        mac.update_padded(additional_data);
        mac.update_padded(&dst[..plaintext.len()]);
        mac.update(&[
            Self::lengths(additional_data.len(), plaintext.len()).into()
        ]);
        tag.copy_from_slice(&mac.finalize());

        Ok(())
    }

    /// Decrypts and authenticates `ciphertext`.
    ///
    /// The result (less the authentication tag) is written to
    /// `dst` and the authentication tag is written to `tag`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least as long as `ciphertext`.
    /// - `ciphertext` must be at most [`C_MAX`][Self::C_MAX] -
    ///   [`TAG_SIZE`][Self::TAG_SIZE] octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    pub fn open_gather(
        &self,
        dst: &mut [u8],
        tag: &[u8; 16],
        nonce: &[u8; 12],
        ciphertext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        if dst.len() < ciphertext.len() {
            return Err(Error::BufferTooSmall);
        }
        if ciphertext.len() as u64 > Self::C_MAX {
            return Err(Error::CiphertextTooLarge);
        }
        if additional_data.len() as u64 > Self::A_MAX {
            return Err(Error::AdditionalDataTooLarge);
        }

        let mut s = Forro::<R>::new(&self.0, nonce);

        let mut mac = {
            let mut poly_key = [0u8; 32];
            s.xor_in_place(&mut poly_key)?;
            Poly1305::new((&poly_key).into())
        };
        mac.update_padded(additional_data);
        mac.update_padded(ciphertext);
        mac.update(&[
            Self::lengths(additional_data.len(), ciphertext.len()).into()
        ]);
        mac.verify(tag.into())?;

        s.xor(&mut dst[..ciphertext.len()], ciphertext)
    }

    fn lengths(ad: usize, pt: usize) -> [u8; 16] {
        let mut out = [0u8; 16];
        LittleEndian::write_u64(&mut out[0..8], ad as u64);
        LittleEndian::write_u64(&mut out[8..16], pt as u64);
        out
    }
}

/// The XForró AEAD using 14 rounds.
pub type XForro14Poly1305 = XForroPoly1305<14>;

/// The XForró AEAD using 10 rounds.
pub type XForro10Poly1305 = XForroPoly1305<10>;

/// The XForró AEAD.
#[cfg_attr(feature = "zeroize", derive(zeroize::ZeroizeOnDrop))]
pub struct XForroPoly1305<const R: usize>([u8; 32]);

impl<const R: usize> XForroPoly1305<R> {
    /// The size in octets of a nonce.
    pub const NONCE_SIZE: usize = 24;

    /// The size in octets of an authentication tag.
    pub const TAG_SIZE: usize = 16;

    /// The maximum size in octets of a plaintext.
    pub const P_MAX: u64 = (1 << 38) - 64;

    /// The maximum size in octets of a ciphertext.
    pub const C_MAX: u64 = Self::P_MAX + Self::TAG_SIZE as u64;

    /// The maximum size in octets of additional data.
    pub const A_MAX: u64 = u64::MAX;

    /// Creates an instance of the AEAD.
    #[inline]
    pub fn new(key: &[u8; 32]) -> Self {
        Self(*key)
    }

    /// Encrypts and authenticates `plaintext`, writing the
    /// result to `dst`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least [`TAG_SIZE`][Self::TAG_SIZE]
    ///   octets longer than `plaintext`.
    /// - `plaintext` must be at most [`P_MAX`][Self::P_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn seal(
        &self,
        dst: &mut [u8],
        nonce: &[u8; 24],
        plaintext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        let (key, nonce) = extend::<R>(&self.0, nonce);
        ForroPoly1305::<R>::new(&key).seal(
            dst,
            &nonce,
            plaintext,
            additional_data,
        )
    }

    /// Decrypts and authenticates `ciphertext`, writing the
    /// result to `dst`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least `ciphertext.len()` -
    ///   [`TAG_SIZE`][Self::TAG_SIZE] bytes long.
    /// - `ciphertext` must be at most [`C_MAX`][Self::C_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn open(
        &self,
        dst: &mut [u8],
        nonce: &[u8; 24],
        ciphertext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        let (key, nonce) = extend::<R>(&self.0, nonce);
        ForroPoly1305::<R>::new(&key).open(
            dst,
            &nonce,
            ciphertext,
            additional_data,
        )
    }

    /// Encrypts and authenticates `plaintext`.
    ///
    /// The result (less the authentication tag) is written to
    /// `dst` and the authentication tag is written to `tag`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least as long as `plaintext`.
    /// - `plaintext` must be at most [`P_MAX`][Self::P_MAX]
    ///   octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn seal_scatter(
        &self,
        dst: &mut [u8],
        tag: &mut [u8; 16],
        nonce: &[u8; 24],
        plaintext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        let (key, nonce) = extend::<R>(&self.0, nonce);
        ForroPoly1305::<R>::new(&key).seal_scatter(
            dst,
            tag,
            &nonce,
            plaintext,
            additional_data,
        )
    }

    /// Decrypts and authenticates `ciphertext`.
    ///
    /// The result (less the authentication tag) is written to
    /// `dst` and the authentication tag is written to `tag`.
    ///
    /// # Requirements
    ///
    /// - `dst` must be at at least as long as `ciphertext`.
    /// - `ciphertext` must be at most [`C_MAX`][Self::C_MAX] -
    ///   [`TAG_SIZE`][Self::TAG_SIZE] octets long.
    /// - `additional_data` must be at most
    ///   [`A_MAX`][Self::A_MAX] octets long.
    #[inline]
    pub fn open_gather(
        &self,
        dst: &mut [u8],
        tag: &[u8; 16],
        nonce: &[u8; 24],
        ciphertext: &[u8],
        additional_data: &[u8],
    ) -> Result<(), Error> {
        let (key, nonce) = extend::<R>(&self.0, nonce);
        ForroPoly1305::<R>::new(&key).open_gather(
            dst,
            tag,
            &nonce,
            ciphertext,
            additional_data,
        )
    }
}

/// Sets `dst = x^y`.
///
/// NB: the lengths are allowed to differ.
#[inline(always)]
fn xor(dst: &mut [u8], x: &[u8], y: &[u8]) {
    for (v, (x, y)) in zip(dst, zip(x, y)) {
        *v = x ^ y;
    }
}

/// Sets `dst ^= src`.
///
/// NB: the lengths are allowed to differ.
#[inline(always)]
fn xor_in_place(dst: &mut [u8], src: &[u8]) {
    for (x, y) in zip(dst, src) {
        *x ^= *y;
    }
}

/// Converts a key to its words.
#[inline(always)]
fn key_to_words(key: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (w, chunk) in zip(out.iter_mut(), key.chunks_exact(4)) {
        *w = LittleEndian::read_u32(chunk);
    }
    out
}

#[inline(always)]
fn extend<const R: usize>(
    key: &[u8; 32],
    nonce: &[u8; 24],
) -> ([u8; 32], [u8; 12]) {
    let mut sub = [0u8; 12];
    sub[4..].copy_from_slice(&nonce[16..24]);
    let key =
        Forro::<R>::hash(&nonce[0..16].try_into().expect("impossible"), key);
    (key, sub)
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "zeroize", derive(zeroize::ZeroizeOnDrop))]
struct State([u32; 16]);

impl State {
    const fn new(key: &[u32; 8], t: &[u32; 2], v: &[u32; 2]) -> Self {
        Self([
            key[0],
            key[1],
            key[2],
            key[3],
            t[0],
            t[1],
            u32::from_le_bytes(*b"volt"),
            u32::from_le_bytes(*b"adaa"),
            key[4],
            key[5],
            key[6],
            key[7],
            v[0],
            v[1],
            u32::from_le_bytes(*b"sabr"),
            u32::from_le_bytes(*b"anca"),
        ])
    }

    #[inline(always)]
    fn core<const R: usize>(&self) -> [u8; 64] {
        let mut x = self.clone();
        x.rounds::<R>();
        let mut dst = [0u8; 64];
        for (chunk, (x, y)) in zip(dst.chunks_exact_mut(4), zip(&x.0, &self.0))
        {
            LittleEndian::write_u32(chunk, x.wrapping_add(*y));
        }
        dst
    }

    /// Performs `R` rounds.
    #[inline(always)]
    fn rounds<const R: usize>(&mut self) {
        for _ in (0..R).step_by(2) {
            self.sr(0, 4, 8, 12, 3);
            self.sr(1, 5, 9, 13, 0);
            self.sr(2, 6, 10, 14, 1);
            self.sr(3, 7, 11, 15, 2);
            self.sr(0, 5, 10, 15, 3);
            self.sr(1, 6, 11, 12, 0);
            self.sr(2, 7, 8, 13, 1);
            self.sr(3, 4, 9, 14, 2);
        }
    }

    /// Performs one subround.
    #[inline(always)]
    fn sr(&mut self, a: usize, b: usize, c: usize, d: usize, e: usize) {
        self[d] = self[d].wrapping_add(self[e]);
        self[c] ^= self[d];
        self[b] = self[b].wrapping_add(self[c]).rotate_left(10);
        self[a] = self[a].wrapping_add(self[b]);
        self[e] ^= self[a];
        self[d] = self[d].wrapping_add(self[e]).rotate_left(27);
        self[c] = self[c].wrapping_add(self[d]);
        self[b] ^= self[c];
        self[a] = self[a].wrapping_add(self[b]).rotate_left(8);
    }

    #[inline(always)]
    fn incr_ctr(&mut self, count64: bool) {
        let (x, c) = self[4].overflowing_add(1);
        self[4] = x;
        if count64 {
            self[5] = self[5].wrapping_add(u32::from(c));
        }
    }
}

impl Deref for State {
    type Target = [u32; 16];

    fn deref(&self) -> &[u32; 16] {
        &self.0
    }
}

impl DerefMut for State {
    fn deref_mut(&mut self) -> &mut [u32; 16] {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forro14_xor_ref1() {
        const MSG: &[u8; 128] = &[0u8; 128];
        const KEY: &[u8; 32] = b"minha vida e andar por este pais";
        const NONCE: &[u8; 8] = b"mostro a";
        const WANT: &[u8; 128] = &[
            0xc5, 0xa9, 0x6c, 0x62, 0xf2, 0x93, 0x52, 0xaf, 0xf2, 0x62, 0x95,
            0xb5, 0x8d, 0xa0, 0x59, 0x5c, 0x62, 0x10, 0x82, 0x25, 0xf1, 0x4e,
            0x33, 0x11, 0x16, 0xad, 0x3f, 0x7b, 0x4e, 0xa0, 0x00, 0xfe, 0xc0,
            0xf0, 0x36, 0x8e, 0x42, 0x11, 0x49, 0xb2, 0x6b, 0x0b, 0x43, 0x98,
            0xdb, 0x7b, 0x3b, 0xbb, 0x99, 0xe3, 0xf5, 0xd7, 0xa9, 0x1b, 0xf0,
            0x28, 0x99, 0x6a, 0x8c, 0x46, 0x51, 0x70, 0x7e, 0xf1, 0xdc, 0xbe,
            0xe0, 0xc1, 0x27, 0x1a, 0x0c, 0xf7, 0xe0, 0x0e, 0xb1, 0xbc, 0x1e,
            0x6f, 0xf8, 0x6e, 0xf2, 0x3c, 0xac, 0xa9, 0x86, 0xa0, 0x03, 0x7e,
            0x02, 0x92, 0x2b, 0xa5, 0xaa, 0x6a, 0x1d, 0x6d, 0xf0, 0x9f, 0x5b,
            0xd1, 0xc5, 0x40, 0xb0, 0xd9, 0xd1, 0xcc, 0x8b, 0x3e, 0xc3, 0x90,
            0x66, 0x0a, 0xe6, 0x8a, 0x88, 0x49, 0xfb, 0x57, 0xea, 0x3a, 0x71,
            0xd8, 0x44, 0xe7, 0x20, 0xb4, 0x84, 0x70,
        ];

        let ciphertext = {
            let mut dst = vec![0u8; MSG.len()];
            Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
                .xor(&mut dst, MSG)
                .expect("should not fail");
            dst
        };
        assert_eq!(ciphertext, WANT);

        let plaintext = {
            let mut dst = vec![0u8; MSG.len()];
            Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
                .xor(&mut dst, &ciphertext)
                .expect("should not fail");
            dst
        };
        assert_eq!(plaintext, MSG);

        let mut data = plaintext;

        Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
            .xor_in_place(&mut data)
            .expect("should not fail");
        assert_eq!(data, WANT);

        Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
            .xor_in_place(&mut data)
            .expect("should not fail");
        assert_eq!(data, MSG);
    }

    #[test]
    fn test_forro14_xor_ref2() {
        const MSG: &[u8; 128] = &[0u8; 128];
        const KEY: &[u8; 32] = b"eu vou mostrar pra voces como se";
        const NONCE: &[u8; 8] = b"danca o ";
        const WANT: &[u8; 128] = &[
            0x4b, 0x76, 0x8c, 0x5c, 0x17, 0x4b, 0xc9, 0xc1, 0xce, 0x1b, 0x8c,
            0x2b, 0x1f, 0xac, 0xe8, 0xe4, 0x5a, 0x63, 0xf9, 0x2e, 0x21, 0xd9,
            0x7b, 0x81, 0xc8, 0x9d, 0x61, 0x90, 0x08, 0x82, 0xd9, 0x27, 0x73,
            0xc5, 0xf7, 0xe6, 0x2a, 0x1f, 0x29, 0x7c, 0xee, 0x9b, 0xae, 0x88,
            0xbb, 0x6c, 0x70, 0x47, 0x7b, 0x80, 0x3a, 0xca, 0xe3, 0x17, 0xc0,
            0x18, 0x46, 0x74, 0xee, 0xfa, 0x43, 0x46, 0x99, 0xb8, 0x50, 0xb6,
            0xa4, 0x5e, 0xd9, 0x7b, 0x34, 0x79, 0x85, 0x2a, 0x76, 0xa6, 0x69,
            0x6a, 0x23, 0x76, 0x9a, 0xaa, 0xc2, 0xd7, 0x35, 0xff, 0x73, 0xf2,
            0x8b, 0x9d, 0xfa, 0x8b, 0x22, 0x42, 0xb2, 0x0b, 0x7c, 0x4e, 0x68,
            0xc0, 0x3d, 0x16, 0x22, 0x6e, 0xe9, 0x06, 0x69, 0x33, 0x59, 0x84,
            0x43, 0xda, 0xf3, 0xbf, 0x43, 0x7b, 0xbc, 0xbc, 0x9f, 0x04, 0xc7,
            0xec, 0xef, 0xa6, 0xa2, 0x4f, 0xad, 0x3d,
        ];

        let ciphertext = {
            let mut dst = vec![0u8; MSG.len()];
            Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
                .xor(&mut dst, MSG)
                .expect("should not fail");
            dst
        };
        assert_eq!(ciphertext, WANT);

        let plaintext = {
            let mut dst = vec![0u8; MSG.len()];
            Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
                .xor(&mut dst, &ciphertext)
                .expect("should not fail");
            dst
        };
        assert_eq!(plaintext, MSG);

        let mut data = plaintext;

        Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
            .xor_in_place(&mut data)
            .expect("should not fail");
        assert_eq!(data, WANT);

        Forro14::new_with_64bit_nonce(KEY, NONCE, 0)
            .xor_in_place(&mut data)
            .expect("should not fail");
        assert_eq!(data, MSG);
    }

    #[test]
    fn test_forro_poly1305_roundtrip() {
        let aead = Forro14Poly1305::new(&[0u8; 32]);

        const NONCE: &[u8; 12] = &[12u8; 12];
        const MSG: &[u8] = b"hello, world!";
        const AD: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

        let (ciphertext, tag) = {
            let mut dst = vec![0u8; MSG.len()];
            let mut tag = [0u8; Forro14Poly1305::TAG_SIZE];
            aead.seal_scatter(&mut dst, &mut tag, NONCE, MSG, AD)
                .expect("should not fail");
            (dst, tag)
        };
        let plaintext = {
            let mut dst = vec![0u8; ciphertext.len()];
            aead.open_gather(&mut dst, &tag, NONCE, &ciphertext, AD)
                .expect("should not fail");
            dst
        };
        assert_eq!(plaintext, MSG);
    }

    #[test]
    fn test_forro_xpoly1305_roundtrip() {
        let aead = XForro14Poly1305::new(&[0u8; 32]);

        const NONCE: &[u8; 24] = &[24u8; 24];
        const MSG: &[u8] = b"hello, world!";
        const AD: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

        let (ciphertext, tag) = {
            let mut dst = vec![0u8; MSG.len()];
            let mut tag = [0u8; XForro14Poly1305::TAG_SIZE];
            aead.seal_scatter(&mut dst, &mut tag, NONCE, MSG, AD)
                .expect("should not fail");
            (dst, tag)
        };
        let plaintext = {
            let mut dst = vec![0u8; ciphertext.len()];
            aead.open_gather(&mut dst, &tag, NONCE, &ciphertext, AD)
                .expect("should not fail");
            dst
        };
        assert_eq!(plaintext, MSG);
    }
}

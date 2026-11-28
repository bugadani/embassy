//! Blocking mutex.
//!
//! This module provides a blocking mutex that can be used to synchronize data.
pub mod raw;

use core::cell::UnsafeCell;

use self::raw::RawMutex;

/// Blocking mutex (not async)
///
/// Provides a blocking mutual exclusion primitive backed by an implementation of [`raw::RawMutex`].
///
/// Which implementation you select depends on the context in which you're using the mutex.
///
/// Use [`CriticalSectionMutex`] when data can be shared between threads and interrupts.
///
/// Use [`NoopMutex`] when data is only shared between tasks running on the same executor.
///
/// The mutex is not reentrant: locking it again while it is locked in the same context panics or
/// deadlocks, depending on the raw mutex. The raw mutexes of this crate panic.
///
/// In all cases, the blocking mutex is intended to be short lived and not held across await points.
/// Use the async [`Mutex`](crate::mutex::Mutex) if you need a lock that is held across await points.
#[derive(Debug)]
pub struct Mutex<R, T: ?Sized> {
    raw: R,
    data: UnsafeCell<T>,
}

unsafe impl<R: Send, T: ?Sized + Send> Send for Mutex<R, T> {}
unsafe impl<R: Sync, T: ?Sized + Send> Sync for Mutex<R, T> {}

impl<R: RawMutex, T> Mutex<R, T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline]
    pub const fn new(val: T) -> Mutex<R, T> {
        Mutex {
            raw: R::INIT,
            data: UnsafeCell::new(val),
        }
    }
}

impl<R: RawMutex, T: ?Sized> Mutex<R, T> {
    /// Locks the mutex and grants temporary mutable access to the protected data.
    ///
    /// # Panics
    ///
    /// Locking the mutex again from `f` panics or deadlocks, depending on the raw mutex. When another
    /// context holds the lock, some raw mutexes panic and others block until it is released.
    pub fn lock<U>(&self, f: impl FnOnce(&mut T) -> U) -> U {
        self.raw.with_lock(|| {
            // Safety: the raw mutex is held, so no other reference to the data exists.
            f(unsafe { &mut *self.data.get() })
        })
    }

    /// Locks the mutex if it is not locked, and grants temporary mutable access to the protected data.
    ///
    /// Returns `None` if the mutex is already locked.
    pub fn try_lock<U>(&self, f: impl FnOnce(&mut T) -> U) -> Option<U> {
        self.raw.try_with_lock(|| {
            // Safety: the raw mutex is held, so no other reference to the data exists.
            f(unsafe { &mut *self.data.get() })
        })
    }

    /// Returns `true` if the mutex is locked.
    pub fn is_locked(&self) -> bool {
        self.raw.is_locked()
    }
}

impl<R, T> Mutex<R, T> {
    /// Creates a new mutex based on a pre-existing raw mutex.
    ///
    /// This allows creating a mutex in a constant context on stable Rust.
    #[inline]
    pub const fn const_new(raw_mutex: R, val: T) -> Mutex<R, T> {
        Mutex {
            raw: raw_mutex,
            data: UnsafeCell::new(val),
        }
    }

    /// Consumes this mutex, returning the underlying data.
    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<R, T: ?Sized> Mutex<R, T> {
    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `Mutex` mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

/// A mutex that allows borrowing data across executors and interrupts.
///
/// # Safety
///
/// This mutex is safe to share between different executors and interrupts.
pub type CriticalSectionMutex<T> = Mutex<raw::CriticalSectionRawMutex, T>;

/// A mutex that panics if locked from multiple contexts.
///
/// # Safety
///
/// This mutex is safe to share between different executors and interrupts.
pub type PanicMutex<T> = Mutex<raw::PanicRawMutex, T>;

/// A mutex that allows borrowing data in the context of a single executor.
///
/// # Safety
///
/// **This Mutex is only safe within a single executor.**
pub type NoopMutex<T> = Mutex<raw::NoopRawMutex, T>;

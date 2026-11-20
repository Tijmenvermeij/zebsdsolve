//! Cancellation crosses native foreign-function threads (including ctypes),
//! even when Zig itself is built single-threaded. The current browser runtime
//! is explicitly wasm32-freestanding with unshared memory and one event loop.
const builtin = @import("builtin");
const browser = builtin.cpu.arch == .wasm32 and builtin.os.tag == .freestanding;

pub const Flag = struct {
    value: bool = false,
    pub fn get(self: *const Flag) bool {
        return if (comptime browser) self.value else @atomicLoad(bool, &self.value, .acquire);
    }
    pub fn set(self: *Flag, value: bool) void {
        if (comptime browser) self.value = value else @atomicStore(bool, &self.value, value, .release);
    }
};

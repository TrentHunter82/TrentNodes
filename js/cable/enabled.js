// The shared flags, alone. A leaf module with no imports, so modules that need
// a flag (bridge, convert, paste, pathing, stitch) depend on THIS instead of
// index.js. Importing index.js made a cycle -- and on frontends where the
// settings store is already hydrated at registerExtension time, onChange fired
// start() synchronously while drag.js was still mid-evaluation on the import
// stack (TDZ crashes on `currentPending`/`finish`, partial installs behind the
// CU-5 sweep-e flake). A leaf breaks the cycle: index.js's body -- and thus
// registerExtension -- cannot run before every module it imports has fully
// evaluated.

let enabled = true;

export function isEnabled() {
  return enabled;
}

export function setEnabled(value) {
  enabled = value;
}

// The `ribbon mode` choice the mixed-modes ruling deferred (pathing.js combsOn):
// outside PCB mode, ribbon runs follow the native link style instead of staying
// trace-drawn. PCB mode ignores this -- traces are the whole point there.
let nativeRibbons = false;

export function nativeRibbonsOn() {
  return nativeRibbons;
}

export function setNativeRibbons(value) {
  nativeRibbons = value;
}

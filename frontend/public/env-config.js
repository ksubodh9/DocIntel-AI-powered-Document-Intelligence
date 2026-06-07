// Fallback: no runtime env injection (non-Docker deployment).
// Docker's nginx entrypoint overwrites this file at container start
// with the actual environment variable values.
window._env_ = window._env_ || {};

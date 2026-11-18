import '@testing-library/jest-dom/vitest';

process.env.NEXT_PUBLIC_RESEARCH_API_URL =
  process.env.NEXT_PUBLIC_RESEARCH_API_URL || 'http://127.0.0.1:8001';

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!globalThis.ResizeObserver) globalThis.ResizeObserver = ResizeObserverStub;

if (!window.matchMedia) {
  window.matchMedia = () => ({
    matches: false,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() { return false; },
  });
}
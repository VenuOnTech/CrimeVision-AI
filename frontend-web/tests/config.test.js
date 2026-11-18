import { describe, it, expect, beforeAll } from 'vitest';

beforeAll(() => {
  // Provide a valid URL before any module imports touch the config.
  process.env.NEXT_PUBLIC_RESEARCH_API_URL = 'http://127.0.0.1:8001';
});

describe('validateResearchApiUrl', () => {
  it('rejects an empty value with an actionable message', async () => {
    const { validateResearchApiUrl } = await import('../src/lib/config');
    expect(() => validateResearchApiUrl('')).toThrow(/NEXT_PUBLIC_RESEARCH_API_URL is not set/i);
  });

  it('rejects a non-URL value', async () => {
    const { validateResearchApiUrl } = await import('../src/lib/config');
    expect(() => validateResearchApiUrl('not-a-url')).toThrow(/not a valid URL/i);
  });

  it('rejects a non-http scheme', async () => {
    const { validateResearchApiUrl } = await import('../src/lib/config');
    expect(() => validateResearchApiUrl('ftp://example.com')).toThrow(/must use http or https/i);
  });

  it('accepts a valid URL and strips a trailing slash', async () => {
    const { validateResearchApiUrl } = await import('../src/lib/config');
    expect(validateResearchApiUrl('http://127.0.0.1:8001/')).toBe('http://127.0.0.1:8001');
  });
});
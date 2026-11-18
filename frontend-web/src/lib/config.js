// Validates the research-engine URL at module load so a misconfigured
// deployment fails loudly rather than silently calling 127.0.0.1.

const RAW_API_URL = process.env.NEXT_PUBLIC_RESEARCH_API_URL;

export function validateResearchApiUrl(rawUrl) {
  if (!rawUrl || !rawUrl.trim()) {
    throw new Error(
      'NEXT_PUBLIC_RESEARCH_API_URL is not set. Create frontend-web/.env.local ' +
        'with NEXT_PUBLIC_RESEARCH_API_URL=http://127.0.0.1:8001 (or the URL of ' +
        'your research engine). See README.md.',
    );
  }
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch (error) {
    throw new Error(`NEXT_PUBLIC_RESEARCH_API_URL is not a valid URL: ${rawUrl}`);
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`NEXT_PUBLIC_RESEARCH_API_URL must use http or https: ${rawUrl}`);
  }
  if (typeof window !== 'undefined') {
    const browserIsLocalhost =
      window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    const apiIsLocalhost =
      parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
    if (browserIsLocalhost !== apiIsLocalhost) {
      // eslint-disable-next-line no-console
      console.warn(
        `[CrimeVision] Browser host (${window.location.hostname}) and ` +
          `research API host (${parsed.hostname}) differ. Make sure the API ` +
          `is reachable from the browser.`,
      );
    }
  }
  return parsed.toString().replace(/\/$/, '');
}

export const RESEARCH_API_URL = validateResearchApiUrl(RAW_API_URL);
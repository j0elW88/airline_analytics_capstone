/**
 * @file src/utils/period.ts
 * @description Helpers for parsing, validating, and sorting period identifiers.
 */

const PERIOD_PATTERN = /(\d{4})_Q([1-4])/i;

export function parsePeriodFromFilename(filename: string): string | null {
  const normalized = filename.replace(/\s+/g, "_");
  const match = PERIOD_PATTERN.exec(normalized);
  if (!match) {
    return null;
  }
  return `${match[1]}_Q${match[2]}`;
}






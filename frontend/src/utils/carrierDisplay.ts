/**
 * @file src/utils/carrierDisplay.ts
 * @description Carrier display formatting helpers and code normalization utilities.
 */

import { CARRIER_NAME_BY_CODE } from "./carriers";

export function normalizeCarrierCode(value: string): string {
  return String(value || "").trim().toUpperCase();
}

export function getCarrierDisplayName(
  code: string,
  lookup: Record<string, string>,
  explicitName?: string,
): string {
  const normalizedCode = normalizeCarrierCode(code);
  const cleanExplicit = String(explicitName || "").trim();
  if (cleanExplicit && normalizeCarrierCode(cleanExplicit) !== normalizedCode) {
    return `${cleanExplicit} (${normalizedCode})`;
  }

  const mapped = lookup[normalizedCode];
  if (mapped) {
    return `${mapped} (${normalizedCode})`;
  }
  const fallbackMapped = CARRIER_NAME_BY_CODE[normalizedCode];
  if (fallbackMapped) {
    return `${fallbackMapped} (${normalizedCode})`;
  }
  return normalizedCode;
}






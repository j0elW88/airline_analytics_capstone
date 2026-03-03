/**
 * @file src/utils/format.ts
 * @description Number, currency, percent, and numeric safety formatting utilities.
 */

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function safeNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

export function toPeriodKey(year: number, quarter: number): string {
  return `${year}_Q${quarter}`;
}

export function parsePeriodLabel(period: string): { year: string; quarter: string } {
  const parts = period.split("_Q");
  if (parts.length !== 2) {
    return { year: "-", quarter: "-" };
  }
  return { year: parts[0], quarter: `Q${parts[1]}` };
}

export function roundToNearest(value: number, nearest: number): number {
  if (nearest <= 0) {
    return value;
  }
  return Math.round(value / nearest) * nearest;
}






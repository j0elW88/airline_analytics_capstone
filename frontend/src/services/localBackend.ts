/**
 * @file src/services/localBackend.ts
 * @description Frontend bridge for local dev API endpoints that read/import backend files.
 */

import type { HubMarketPowerRow, RouteMarketPowerRow } from "../types/data";

interface LocalPeriodsResponse {
  periods: string[];
}

interface CarrierLookupResponse {
  carriers: Record<string, string>;
}

interface LocalDatasetResponse {
  period: string;
  routeRows: RouteMarketPowerRow[];
  hubRows: HubMarketPowerRow[];
}

interface ImportRawResponse {
  period: string;
}

export interface FareDistributionBin {
  fareStart: number;
  fareEnd: number;
  passengers: number;
  rowCount: number;
}

export interface FareDistributionCarrier {
  carrier: string;
  carrierName: string;
  totalPassengers: number;
  totalRows: number;
  minFare: number;
  maxFare: number;
  bins: FareDistributionBin[];
}

export interface RouteFareDistributionResponse {
  period: string;
  origin: string;
  dest: string;
  carrierFilter: string;
  carriers: FareDistributionCarrier[];
}

const routeFareDistributionSessionCache = new Map<string, RouteFareDistributionResponse>();

export class ImportRawError extends Error {
  readonly errorType: "verification" | "execution" | "unknown";
  readonly stage: "parse" | "analyze" | "unknown";

  constructor(
    message: string,
    errorType: "verification" | "execution" | "unknown",
    stage: "parse" | "analyze" | "unknown",
  ) {
    super(message);
    this.name = "ImportRawError";
    this.errorType = errorType;
    this.stage = stage;
  }
}

export async function fetchLocalPeriods(): Promise<string[]> {
  // Reads available analyzed periods from Vite local API bridge.
  const response = await fetch("/api/local/periods");
  if (!response.ok) {
    throw new Error(`Failed reading local periods (${response.status}).`);
  }

  const data = (await response.json()) as LocalPeriodsResponse;
  if (!Array.isArray(data.periods)) {
    return [];
  }

  return data.periods;
}

export async function fetchCarrierLookup(): Promise<Record<string, string>> {
  // Pulls backend carrier-code translation map for user-friendly labels.
  const response = await fetch("/api/local/carriers");
  if (!response.ok) {
    throw new Error(`Failed reading carrier lookup (${response.status}).`);
  }

  const data = (await response.json()) as CarrierLookupResponse;
  if (!data.carriers || typeof data.carriers !== "object") {
    return {};
  }

  return data.carriers;
}

export async function fetchLocalDataset(period: string): Promise<{
  period: string;
  routeRows: RouteMarketPowerRow[];
  hubRows: HubMarketPowerRow[];
}> {
  // Returns already-analyzed route/hub row payloads for one selected period.
  const response = await fetch(`/api/local/dataset?period=${encodeURIComponent(period)}`);
  if (!response.ok) {
    throw new Error(`Failed loading local dataset for ${period} (${response.status}).`);
  }

  const data = (await response.json()) as LocalDatasetResponse;
  const routeRows = Array.isArray(data.routeRows) ? data.routeRows : [];
  const hubRows = Array.isArray(data.hubRows) ? data.hubRows : [];

  return {
    period: data.period,
    routeRows,
    hubRows,
  };
}

export async function fetchRouteFareDistribution(params: {
  period: string;
  origin: string;
  dest: string;
  carrier?: string;
}): Promise<RouteFareDistributionResponse> {
  const period = params.period.trim();
  const origin = params.origin.trim().toUpperCase();
  const dest = params.dest.trim().toUpperCase();
  const carrier = (params.carrier ?? "").trim().toUpperCase();
  const key = `${period}|${origin}|${dest}|${carrier}`;
  const cached = routeFareDistributionSessionCache.get(key);
  if (cached) {
    return cached;
  }

  const query = new URLSearchParams({
    period,
    origin,
    dest,
  });
  if (carrier) {
    query.set("carrier", carrier);
  }
  const response = await fetch(`/api/local/fare-distribution?${query.toString()}`);
  if (!response.ok) {
    let message = `Failed loading fare distribution (${response.status}).`;
    try {
      const payload = (await response.json()) as { error?: string };
      if (payload.error) {
        message = payload.error;
      }
    } catch {
      // keep fallback message
    }
    throw new Error(message);
  }

  const data = (await response.json()) as RouteFareDistributionResponse;
  if (!Array.isArray(data.carriers)) {
    throw new Error("Fare distribution payload missing carriers.");
  }
  routeFareDistributionSessionCache.set(key, data);
  return data;
}

export async function importRawDb1b(file: File): Promise<string> {
  // Uploads a raw DB1B file to bridge endpoint that runs parse + analyze scripts.
  const response = await fetch(`/api/local/import-raw?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
    },
    body: file,
  });

  if (!response.ok) {
    // Bubble backend/import errors up to UI modal as human-readable text.
    let message = "Raw import failed.";
    let errorType: "verification" | "execution" | "unknown" = "unknown";
    let stage: "parse" | "analyze" | "unknown" = "unknown";
    try {
      const payload = (await response.json()) as {
        error?: string;
        errorType?: "verification" | "execution";
        stage?: "parse" | "analyze";
      };
      if (payload.error) {
        message = payload.error;
      }
      if (payload.errorType === "verification" || payload.errorType === "execution") {
        errorType = payload.errorType;
      }
      if (payload.stage === "parse" || payload.stage === "analyze") {
        stage = payload.stage;
      }
    } catch {
      // Keep fallback message.
    }
    throw new ImportRawError(message, errorType, stage);
  }

  const data = (await response.json()) as ImportRawResponse;
  if (!data.period) {
    throw new Error("Import completed but period could not be determined.");
  }

  return data.period;
}






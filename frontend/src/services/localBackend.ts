/**
 * @file src/services/localBackend.ts
 * @description Frontend bridge for local dev API endpoints that read/import backend files.
 */

import type { HubMarketPowerRow, RouteMarketPowerRow } from "../types/data";
import { parseHubMarketPowerCsv, parseRouteMarketPowerCsv } from "./csvParser";

interface LocalPeriodsResponse {
  periods: string[];
}

interface CarrierLookupResponse {
  carriers: Record<string, string>;
}

interface LocalDatasetResponse {
  period: string;
  routeCsv: string;
  hubCsv: string;
}

interface ImportRawResponse {
  period: string;
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
  // Returns already-analyzed route/hub csv payloads for one selected period.
  const response = await fetch(`/api/local/dataset?period=${encodeURIComponent(period)}`);
  if (!response.ok) {
    throw new Error(`Failed loading local dataset for ${period} (${response.status}).`);
  }

  const data = (await response.json()) as LocalDatasetResponse;
  const routeRows = parseRouteMarketPowerCsv(data.routeCsv);
  const hubRows = parseHubMarketPowerCsv(data.hubCsv);

  return {
    period: data.period,
    routeRows,
    hubRows,
  };
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
    try {
      const payload = (await response.json()) as { error?: string };
      if (payload.error) {
        message = payload.error;
      }
    } catch {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  const data = (await response.json()) as ImportRawResponse;
  if (!data.period) {
    throw new Error("Import completed but period could not be determined.");
  }

  return data.period;
}






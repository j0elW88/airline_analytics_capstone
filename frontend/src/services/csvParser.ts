/**
 * @file src/services/csvParser.ts
 * @description CSV parsing and validation helpers for route/hub market power files.
 */

import type { HubMarketPowerRow, RouteMarketPowerRow } from "../types/data";
import { safeNumber } from "../utils/format";

function splitCsvLine(line: string): string[] {
  // Minimal CSV tokenizer that supports quoted fields and escaped quotes.
  const out: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      out.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  out.push(current.trim());
  return out;
}

function parseCsvRows(csvText: string): Record<string, string>[] {
  // Normalizes line endings, skips blank lines, and maps each line to a header/value object.
  const lines = csvText
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter((line) => line.trim().length > 0);

  if (lines.length < 2) {
    return [];
  }

  const headers = splitCsvLine(lines[0]);
  const rows: Record<string, string>[] = [];

  for (let i = 1; i < lines.length; i += 1) {
    const values = splitCsvLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx] ?? "";
    });
    rows.push(row);
  }

  return rows;
}

function assertColumns(rows: Record<string, string>[], required: string[], label: string): void {
  // Guardrail that surfaces friendly errors when an uploaded file has wrong structure.
  if (rows.length === 0) {
    throw new Error(`${label} file is empty.`);
  }
  const missing = required.filter((column) => !(column in rows[0]));
  if (missing.length > 0) {
    throw new Error(`${label} file missing required columns: ${missing.join(", ")}`);
  }
}

export async function readFileText(file: File): Promise<string> {
  // Promise wrapper around FileReader for async/await usage.
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Failed reading file: ${file.name}`));
    reader.readAsText(file);
  });
}

export function parseRouteMarketPowerCsv(csvText: string): RouteMarketPowerRow[] {
  // Converts raw route market power CSV text into strongly-typed numeric objects.
  const rows = parseCsvRows(csvText);
  assertColumns(
    rows,
    ["Origin", "Dest", "Carrier", "avg_fare_weighted", "avg_distance_weighted", "total_passengers"],
    "Route market power",
  );

  return rows.map((row) => ({
    Origin: row.Origin,
    Dest: row.Dest,
    Carrier: row.Carrier,
    carrier_name: row.carrier_name || undefined,
    OriginState: row.OriginState || undefined,
    total_passengers: safeNumber(row.total_passengers),
    row_count: safeNumber(row.row_count),
    avg_fare_weighted: safeNumber(row.avg_fare_weighted),
    avg_distance_weighted: safeNumber(row.avg_distance_weighted),
    route_total_passengers_all: safeNumber(row.route_total_passengers_all),
    route_total_passengers_valid: safeNumber(row.route_total_passengers_valid),
    carriers_on_route_all: safeNumber(row.carriers_on_route_all),
    carriers_on_route_valid: safeNumber(row.carriers_on_route_valid),
    route_share: safeNumber(row.route_share),
    route_HHI: safeNumber(row.route_HHI),
    route_avg_fare_all: safeNumber(row.route_avg_fare_all),
    route_min_fare_all: safeNumber(row.route_min_fare_all),
  }));
}

export function parseHubMarketPowerCsv(csvText: string): HubMarketPowerRow[] {
  // Converts raw hub market power CSV text into strongly-typed numeric objects.
  const rows = parseCsvRows(csvText);
  assertColumns(
    rows,
    ["Origin", "OriginState", "Carrier", "avg_fare_weighted", "avg_distance_weighted", "total_passengers"],
    "Hub market power",
  );

  return rows.map((row) => ({
    Origin: row.Origin,
    OriginState: row.OriginState,
    Carrier: row.Carrier,
    carrier_name: row.carrier_name || undefined,
    total_passengers: safeNumber(row.total_passengers),
    row_count: safeNumber(row.row_count),
    avg_fare_weighted: safeNumber(row.avg_fare_weighted),
    avg_distance_weighted: safeNumber(row.avg_distance_weighted),
    hub_total_passengers_all: safeNumber(row.hub_total_passengers_all),
    hub_total_passengers_valid: safeNumber(row.hub_total_passengers_valid),
    carriers_at_hub_all: safeNumber(row.carriers_at_hub_all),
    carriers_at_hub_valid: safeNumber(row.carriers_at_hub_valid),
    hub_share: safeNumber(row.hub_share),
    hub_HHI: safeNumber(row.hub_HHI),
    hub_avg_fare_all: safeNumber(row.hub_avg_fare_all),
    hub_min_fare_all: safeNumber(row.hub_min_fare_all),
  }));
}






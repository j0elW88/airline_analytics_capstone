import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BACKEND_ROOT = path.resolve(__dirname, "../backend");
const ROUTE_DIR = path.join(BACKEND_ROOT, "routeMP_folder");
const HUB_DIR = path.join(BACKEND_ROOT, "hubMP_folder");
const FARE_DISTRIBUTION_DIR = path.join(BACKEND_ROOT, "specific_fare_distribution_charts");
const UPLOADS_DIR = path.join(BACKEND_ROOT, "uploads");

const ROUTE_PATTERN = /^route_market_power_(\d{4}_Q[1-4])\.parquet$/i;
const HUB_PATTERN = /^hub_market_power_(\d{4}_Q[1-4])\.parquet$/i;
const FARE_LOWER_BOUND = 50;
const FARE_UPPER_BOUND = 1200;
const VERIFICATION_FAILURE_PATTERNS = [
  /missing required columns/i,
  /could not detect year\/quarter/i,
  /multiple year\/quarter values/i,
  /missing file:/i,
  /filenotfounderror/i,
  /valueerror/i,
  /could not find raw file/i,
];

interface FareDistributionRow {
  Origin: string;
  Dest: string;
  Carrier: string;
  fare_bin_start: number;
  fare_bin_end: number;
  passengers_sum: number;
  row_count: number;
}

interface FareDistributionBinPayload {
  fareStart: number;
  fareEnd: number;
  passengers: number;
  rowCount: number;
}

interface FareDistributionCarrierPayload {
  carrier: string;
  carrierName: string;
  totalPassengers: number;
  totalRows: number;
  minFare: number;
  maxFare: number;
  bins: FareDistributionBinPayload[];
}

interface FareDistributionResponse {
  period: string;
  origin: string;
  dest: string;
  carrierFilter: string;
  carriers: FareDistributionCarrierPayload[];
}

interface HubFareDistributionResponse {
  period: string;
  originScope: string;
  carrierFilter: string;
  carriers: FareDistributionCarrierPayload[];
}

const fareDistributionPeriodCache = new Map<string, FareDistributionRow[]>();
const fareDistributionSessionCache = new Map<string, FareDistributionResponse>();
const hubFareDistributionSessionCache = new Map<string, HubFareDistributionResponse>();

function json(res: any, status: number, payload: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

async function readRequestBody(req: any): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

function sanitizeFilename(input: string): string {
  const base = path.basename(input);
  return base.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function isBackendVerificationFailure(output: string): boolean {
  const text = String(output ?? "");
  return VERIFICATION_FAILURE_PATTERNS.some((pattern) => pattern.test(text));
}

function runCommand(command: string, args: string[], cwd: string): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, {
      cwd,
      shell: false,
      windowsHide: true,
    });
    let output = "";

    proc.stdout.on("data", (chunk) => {
      output += String(chunk);
    });
    proc.stderr.on("data", (chunk) => {
      output += String(chunk);
    });
    proc.on("close", (code) => {
      resolve({ code: code ?? 1, output });
    });
    proc.on("error", (error) => {
      resolve({
        code: 1,
        output: `${output}\n${String(error?.message ?? error)}`.trim(),
      });
    });
  });
}

async function collectCompletePeriods(): Promise<string[]> {
  const [routeFiles, hubFiles] = await Promise.all([
    fs.readdir(ROUTE_DIR),
    fs.readdir(HUB_DIR),
  ]);

  const routePeriods = new Set<string>();
  for (const file of routeFiles) {
    const match = ROUTE_PATTERN.exec(file);
    if (match) {
      routePeriods.add(match[1]);
    }
  }

  const completePeriods: string[] = [];
  for (const file of hubFiles) {
    const match = HUB_PATTERN.exec(file);
    if (!match) {
      continue;
    }
    const period = match[1];
    if (routePeriods.has(period)) {
      completePeriods.push(period);
    }
  }

  return completePeriods.sort();
}

async function collectCarrierLookup(): Promise<Record<string, string>> {
  const result = await runCommand(
    "py",
    ["-c", "import json,carrier_codes;print(json.dumps(carrier_codes.CARRIER_LOOKUP))"],
    BACKEND_ROOT,
  );
  if (result.code !== 0) {
    throw new Error(`Failed loading carrier lookup from backend:\n${result.output}`);
  }

  const start = result.output.indexOf("{");
  const end = result.output.lastIndexOf("}");
  if (start < 0 || end < 0 || end <= start) {
    throw new Error("Carrier lookup response was not valid JSON.");
  }

  const jsonPayload = result.output.slice(start, end + 1);
  const parsed = JSON.parse(jsonPayload) as Record<string, string>;
  const normalized: Record<string, string> = {};
  Object.entries(parsed).forEach(([code, name]) => {
    const key = String(code || "").trim().toUpperCase();
    if (!key) {
      return;
    }
    normalized[key] = String(name || "").trim();
  });
  return normalized;
}

function parsePeriod(period: string): { year: string; quarter: string } | null {
  const match = /^(\d{4})_Q([1-4])$/i.exec(period.trim());
  if (!match) {
    return null;
  }
  return { year: match[1], quarter: match[2] };
}

function parseJsonArrayOutput(raw: string): Record<string, unknown>[] {
  const text = String(raw ?? "").trim();
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start < 0 || end < 0 || end <= start) {
    throw new Error("Backend Python output did not contain a JSON array payload.");
  }
  const payload = text.slice(start, end + 1);
  const parsed = JSON.parse(payload) as unknown;
  if (!Array.isArray(parsed)) {
    throw new Error("Expected JSON array payload.");
  }
  return parsed as Record<string, unknown>[];
}

async function readParquetRows(fullPath: string): Promise<Record<string, unknown>[]> {
  const result = await runCommand(
    "py",
    [
      "-c",
      "import json,pandas as pd,sys;print(pd.read_parquet(sys.argv[1]).to_json(orient='records'))",
      fullPath,
    ],
    BACKEND_ROOT,
  );
  if (result.code !== 0) {
    throw new Error(`Failed reading parquet file: ${fullPath}\n${result.output}`);
  }
  return parseJsonArrayOutput(result.output);
}

function normalizeFareDistributionRows(rawRows: Record<string, unknown>[]): FareDistributionRow[] {
  if (rawRows.length === 0) {
    return [];
  }
  const rows: FareDistributionRow[] = [];
  for (const row of rawRows) {
    rows.push({
      Origin: String(row.Origin ?? "").trim().toUpperCase(),
      Dest: String(row.Dest ?? "").trim().toUpperCase(),
      Carrier: String(row.Carrier ?? "").trim().toUpperCase(),
      fare_bin_start: Number(row.fare_bin_start ?? 0),
      fare_bin_end: Number(row.fare_bin_end ?? 0),
      passengers_sum: Number(row.passengers_sum ?? 0),
      row_count: Number(row.row_count ?? 0),
    });
  }
  return rows;
}

async function ensureFareDistributionCache(period: string): Promise<void> {
  const filename = `specific_fare_distribution_${period}.parquet`;
  const fullPath = path.join(FARE_DISTRIBUTION_DIR, filename);
  try {
    await fs.access(fullPath);
    return;
  } catch {
    // Cache missing: regenerate for this period from raw uploads.
  }

  const parsed = parsePeriod(period);
  if (!parsed) {
    throw new Error(`Invalid period format: ${period}`);
  }

  const parseResult = await runCommand(
    "py",
    [
      "capstone_parse.py",
      "--year",
      parsed.year,
      "--quarter",
      parsed.quarter,
      "--verbose",
      "0",
    ],
    BACKEND_ROOT,
  );
  if (parseResult.code !== 0) {
    throw new Error(`Failed generating fare distribution cache for ${period}:\n${parseResult.output}`);
  }
}

async function loadFareDistributionRows(period: string): Promise<FareDistributionRow[]> {
  if (fareDistributionPeriodCache.has(period)) {
    return fareDistributionPeriodCache.get(period) ?? [];
  }
  await ensureFareDistributionCache(period);
  const filename = `specific_fare_distribution_${period}.parquet`;
  const fullPath = path.join(FARE_DISTRIBUTION_DIR, filename);
  const rawRows = await readParquetRows(fullPath);
  const rows = normalizeFareDistributionRows(rawRows);
  fareDistributionPeriodCache.set(period, rows);
  return rows;
}

function buildFareDistributionResponse(
  period: string,
  origin: string,
  dest: string,
  carrierFilter: string,
  rows: FareDistributionRow[],
  carrierLookup: Record<string, string>,
): FareDistributionResponse {
  const byCarrier = new Map<string, {
    totalPassengers: number;
    totalRows: number;
    minFare: number;
    maxFare: number;
    bins: Map<string, FareDistributionBinPayload>;
  }>();

  rows.forEach((row) => {
    if (row.Origin !== origin || row.Dest !== dest) {
      return;
    }
    if (carrierFilter && row.Carrier !== carrierFilter) {
      return;
    }
    const boundedStart = Math.max(row.fare_bin_start, FARE_LOWER_BOUND);
    const boundedEnd = Math.min(row.fare_bin_end, FARE_UPPER_BOUND);
    if (!Number.isFinite(boundedStart) || !Number.isFinite(boundedEnd) || boundedEnd <= boundedStart) {
      return;
    }
    const carrier = row.Carrier;
    const current = byCarrier.get(carrier) ?? {
      totalPassengers: 0,
      totalRows: 0,
      minFare: Number.POSITIVE_INFINITY,
      maxFare: Number.NEGATIVE_INFINITY,
      bins: new Map<string, FareDistributionBinPayload>(),
    };
    current.totalPassengers += row.passengers_sum;
    current.totalRows += row.row_count;
    current.minFare = Math.min(current.minFare, boundedStart);
    current.maxFare = Math.max(current.maxFare, boundedEnd);
    const binKey = `${boundedStart}|${boundedEnd}`;
    const existingBin = current.bins.get(binKey) ?? {
      fareStart: boundedStart,
      fareEnd: boundedEnd,
      passengers: 0,
      rowCount: 0,
    };
    existingBin.passengers += row.passengers_sum;
    existingBin.rowCount += row.row_count;
    current.bins.set(binKey, existingBin);
    byCarrier.set(carrier, current);
  });

  const carriers = Array.from(byCarrier.entries())
    .map(([carrier, agg]) => ({
      carrier,
      carrierName: carrierLookup[carrier] || carrier,
      totalPassengers: agg.totalPassengers,
      totalRows: agg.totalRows,
      minFare: Number.isFinite(agg.minFare) ? agg.minFare : 0,
      maxFare: Number.isFinite(agg.maxFare) ? agg.maxFare : 0,
      bins: Array.from(agg.bins.values()).sort((a, b) => a.fareStart - b.fareStart),
    }))
    .sort((a, b) => b.totalPassengers - a.totalPassengers);

  return {
    period,
    origin,
    dest,
    carrierFilter,
    carriers,
  };
}

function buildHubFareDistributionResponse(
  period: string,
  originScope: string,
  carrierFilter: string,
  rows: FareDistributionRow[],
  carrierLookup: Record<string, string>,
): HubFareDistributionResponse {
  const normalizedOriginScope = originScope.trim().toUpperCase();
  const byCarrier = new Map<string, {
    totalPassengers: number;
    totalRows: number;
    minFare: number;
    maxFare: number;
    bins: Map<string, FareDistributionBinPayload>;
  }>();

  rows.forEach((row) => {
    if (normalizedOriginScope && row.Origin !== normalizedOriginScope) {
      return;
    }
    if (carrierFilter && row.Carrier !== carrierFilter) {
      return;
    }
    const boundedStart = Math.max(row.fare_bin_start, FARE_LOWER_BOUND);
    const boundedEnd = Math.min(row.fare_bin_end, FARE_UPPER_BOUND);
    if (!Number.isFinite(boundedStart) || !Number.isFinite(boundedEnd) || boundedEnd <= boundedStart) {
      return;
    }
    const carrier = row.Carrier;
    const current = byCarrier.get(carrier) ?? {
      totalPassengers: 0,
      totalRows: 0,
      minFare: Number.POSITIVE_INFINITY,
      maxFare: Number.NEGATIVE_INFINITY,
      bins: new Map<string, FareDistributionBinPayload>(),
    };
    current.totalPassengers += row.passengers_sum;
    current.totalRows += row.row_count;
    current.minFare = Math.min(current.minFare, boundedStart);
    current.maxFare = Math.max(current.maxFare, boundedEnd);
    const binKey = `${boundedStart}|${boundedEnd}`;
    const existingBin = current.bins.get(binKey) ?? {
      fareStart: boundedStart,
      fareEnd: boundedEnd,
      passengers: 0,
      rowCount: 0,
    };
    existingBin.passengers += row.passengers_sum;
    existingBin.rowCount += row.row_count;
    current.bins.set(binKey, existingBin);
    byCarrier.set(carrier, current);
  });

  const carriers = Array.from(byCarrier.entries())
    .map(([carrier, agg]) => ({
      carrier,
      carrierName: carrierLookup[carrier] || carrier,
      totalPassengers: agg.totalPassengers,
      totalRows: agg.totalRows,
      minFare: Number.isFinite(agg.minFare) ? agg.minFare : 0,
      maxFare: Number.isFinite(agg.maxFare) ? agg.maxFare : 0,
      bins: Array.from(agg.bins.values()).sort((a, b) => a.fareStart - b.fareStart),
    }))
    .sort((a, b) => b.totalPassengers - a.totalPassengers);

  return {
    period,
    originScope: normalizedOriginScope || "ALL",
    carrierFilter,
    carriers,
  };
}

function localDataPlugin(): Plugin {
  return {
    name: "local-backend-data",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url) {
          next();
          return;
        }

        const requestUrl = new URL(req.url, "http://localhost");

        if (requestUrl.pathname === "/api/local/periods") {
          try {
            const periods = await collectCompletePeriods();
            json(res, 200, { periods });
          } catch (error) {
            const message = error instanceof Error ? error.message : "Failed reading local periods.";
            json(res, 500, { error: message, periods: [] });
          }
          return;
        }

        if (requestUrl.pathname === "/api/local/carriers") {
          try {
            const carriers = await collectCarrierLookup();
            json(res, 200, { carriers });
          } catch (error) {
            const message = error instanceof Error ? error.message : "Failed reading carrier lookup.";
            json(res, 500, { error: message, carriers: {} });
          }
          return;
        }

        if (requestUrl.pathname === "/api/local/fare-distribution") {
          const period = requestUrl.searchParams.get("period")?.trim() ?? "";
          const origin = requestUrl.searchParams.get("origin")?.trim().toUpperCase() ?? "";
          const dest = requestUrl.searchParams.get("dest")?.trim().toUpperCase() ?? "";
          const carrier = requestUrl.searchParams.get("carrier")?.trim().toUpperCase() ?? "";

          if (!period || !origin || !dest) {
            json(res, 400, { error: "Missing required query params: period, origin, dest." });
            return;
          }

          const sessionKey = `${period}|${origin}|${dest}|${carrier}`;
          const cached = fareDistributionSessionCache.get(sessionKey);
          if (cached) {
            json(res, 200, cached);
            return;
          }

          try {
            const [rows, carrierLookup] = await Promise.all([
              loadFareDistributionRows(period),
              collectCarrierLookup(),
            ]);
            const payload = buildFareDistributionResponse(period, origin, dest, carrier, rows, carrierLookup);
            if (payload.carriers.length === 0) {
              json(res, 404, {
                error: `No fare distribution data found for ${period} ${origin}->${dest}${carrier ? ` (${carrier})` : ""}.`,
              });
              return;
            }
            fareDistributionSessionCache.set(sessionKey, payload);
            json(res, 200, payload);
          } catch (error) {
            const message = error instanceof Error ? error.message : "Failed loading fare distribution.";
            json(res, 500, { error: message });
          }
          return;
        }

        if (requestUrl.pathname === "/api/local/hub-fare-distribution") {
          const period = requestUrl.searchParams.get("period")?.trim() ?? "";
          const origin = requestUrl.searchParams.get("origin")?.trim().toUpperCase() ?? "";
          const carrier = requestUrl.searchParams.get("carrier")?.trim().toUpperCase() ?? "";

          if (!period) {
            json(res, 400, { error: "Missing required query param: period." });
            return;
          }

          const sessionKey = `${period}|${origin || "ALL"}|${carrier}`;
          const cached = hubFareDistributionSessionCache.get(sessionKey);
          if (cached) {
            json(res, 200, cached);
            return;
          }

          try {
            const [rows, carrierLookup] = await Promise.all([
              loadFareDistributionRows(period),
              collectCarrierLookup(),
            ]);
            const payload = buildHubFareDistributionResponse(period, origin, carrier, rows, carrierLookup);
            if (payload.carriers.length === 0) {
              json(res, 404, {
                error: `No hub fare distribution data found for ${period}${origin ? ` ${origin}` : ""}${carrier ? ` (${carrier})` : ""}.`,
              });
              return;
            }
            hubFareDistributionSessionCache.set(sessionKey, payload);
            json(res, 200, payload);
          } catch (error) {
            const message = error instanceof Error ? error.message : "Failed loading hub fare distribution.";
            json(res, 500, { error: message });
          }
          return;
        }

        if (requestUrl.pathname === "/api/local/dataset") {
          const period = requestUrl.searchParams.get("period")?.trim();
          if (!period) {
            json(res, 400, { error: "Missing period query param." });
            return;
          }

          try {
            const routePath = path.join(ROUTE_DIR, `route_market_power_${period}.parquet`);
            const hubPath = path.join(HUB_DIR, `hub_market_power_${period}.parquet`);
            const [routeRows, hubRows] = await Promise.all([
              readParquetRows(routePath),
              readParquetRows(hubPath),
            ]);

            json(res, 200, { period, routeRows, hubRows });
          } catch (error) {
            const message = error instanceof Error ? error.message : `Failed loading dataset for ${period}.`;
            json(res, 404, { error: message });
          }
          return;
        }

        if (requestUrl.pathname === "/api/local/import-raw") {
          if (req.method !== "POST") {
            json(res, 405, { error: "Method not allowed. Use POST." });
            return;
          }

          try {
            const fileBuffer = await readRequestBody(req);
            if (!fileBuffer || fileBuffer.length === 0) {
              json(res, 400, { error: "Uploaded file is empty." });
              return;
            }

            const requestedFilename = requestUrl.searchParams.get("filename") || `db1b_upload_${Date.now()}.csv`;
            const safeFilename = sanitizeFilename(requestedFilename);
            await fs.mkdir(UPLOADS_DIR, { recursive: true });
            const uploadPath = path.join(UPLOADS_DIR, safeFilename);
            await fs.writeFile(uploadPath, fileBuffer);

            const parseResult = await runCommand(
              "py",
              ["capstone_parse.py", "--csv", uploadPath, "--verbose", "0", "--delete_raw_csv"],
              BACKEND_ROOT,
            );
            if (parseResult.code !== 0) {
              const errorType = isBackendVerificationFailure(parseResult.output) ? "verification" : "execution";
              const status = errorType === "verification" ? 422 : 400;
              json(res, status, {
                error: `capstone_parse failed:\n${parseResult.output}`,
                errorType,
                stage: "parse",
              });
              return;
            }

            const periodMatch = /period used:\s*Year=(\d+),\s*Quarter=(\d+)/i.exec(parseResult.output);
            if (!periodMatch) {
              json(res, 500, {
                error: "Parse succeeded but could not detect output Year/Quarter from parser logs.",
              });
              return;
            }

            const year = periodMatch[1];
            const quarter = periodMatch[2];
            const period = `${year}_Q${quarter}`;

            const analyzeResult = await runCommand(
              "py",
              [
                "capstone_analyze.py",
                "--year",
                year,
                "--quarter",
                quarter,
                "--dir",
                ".",
                "--export_parquet",
                "--verbose",
                "0",
              ],
              BACKEND_ROOT,
            );
            if (analyzeResult.code !== 0) {
              const errorType = isBackendVerificationFailure(analyzeResult.output) ? "verification" : "execution";
              const status = errorType === "verification" ? 422 : 500;
              json(res, status, {
                error: `capstone_analyze failed:\n${analyzeResult.output}`,
                errorType,
                stage: "analyze",
              });
              return;
            }

            fareDistributionPeriodCache.delete(period);
            for (const key of Array.from(fareDistributionSessionCache.keys())) {
              if (key.startsWith(`${period}|`)) {
                fareDistributionSessionCache.delete(key);
              }
            }
            for (const key of Array.from(hubFareDistributionSessionCache.keys())) {
              if (key.startsWith(`${period}|`)) {
                hubFareDistributionSessionCache.delete(key);
              }
            }

            json(res, 200, { period });
          } catch (error) {
            const message = error instanceof Error ? error.message : "Raw import failed unexpectedly.";
            json(res, 500, { error: message });
          }
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localDataPlugin()],
});

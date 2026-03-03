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
const UPLOADS_DIR = path.join(BACKEND_ROOT, "uploads");

const ROUTE_PATTERN = /^route_market_power_(\d{4}_Q[1-4])\.csv$/i;
const HUB_PATTERN = /^hub_market_power_(\d{4}_Q[1-4])\.csv$/i;
const VERIFICATION_FAILURE_PATTERNS = [
  /missing required columns/i,
  /could not detect year\/quarter/i,
  /multiple year\/quarter values/i,
  /missing file:/i,
  /filenotfounderror/i,
  /valueerror/i,
  /could not find raw file/i,
];

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

        if (requestUrl.pathname === "/api/local/dataset") {
          const period = requestUrl.searchParams.get("period")?.trim();
          if (!period) {
            json(res, 400, { error: "Missing period query param." });
            return;
          }

          try {
            const routePath = path.join(ROUTE_DIR, `route_market_power_${period}.csv`);
            const hubPath = path.join(HUB_DIR, `hub_market_power_${period}.csv`);
            const [routeCsv, hubCsv] = await Promise.all([
              fs.readFile(routePath, "utf-8"),
              fs.readFile(hubPath, "utf-8"),
            ]);

            json(res, 200, { period, routeCsv, hubCsv });
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
              ["capstone_parse.py", "--csv", uploadPath, "--verbose", "0"],
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
                "--export_csv",
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

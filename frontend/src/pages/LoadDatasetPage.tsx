/**
 * @file src/pages/LoadDatasetPage.tsx
 * @description Screen for importing raw DB1B files or loading existing backend periods.
 */

import { useEffect, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";
import { fetchLocalPeriods } from "../services/localBackend";

interface LoadDatasetPageProps {
  onImportRaw: (file: File) => Promise<void>;
  onImportExisting: (period: string) => Promise<void>;
  onImportFailed: (message: string) => void;
}

export function LoadDatasetPage({
  onImportRaw,
  onImportExisting,
  onImportFailed,
}: LoadDatasetPageProps) {
  // Raw DB1B upload state.
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [isProcessingRaw, setIsProcessingRaw] = useState(false);
  const [rawElapsedSeconds, setRawElapsedSeconds] = useState(0);

  // Existing period import state (backend-generated route/hub outputs).
  const [localPeriods, setLocalPeriods] = useState<string[]>([]);
  const [selectedLocalPeriod, setSelectedLocalPeriod] = useState("");
  const [isLoadingExisting, setIsLoadingExisting] = useState(false);

  useEffect(() => {
    if (!isProcessingRaw) {
      setRawElapsedSeconds(0);
      return;
    }
    const timer = window.setInterval(() => {
      setRawElapsedSeconds((previous) => previous + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isProcessingRaw]);

  useEffect(() => {
    let active = true;
    async function loadPeriods() {
      try {
        // Fetch all complete periods currently available from local backend folders.
        const periods = await fetchLocalPeriods();
        if (!active) {
          return;
        }
        setLocalPeriods(periods);
        if (periods.length > 0) {
          setSelectedLocalPeriod((prev) => (prev && periods.includes(prev) ? prev : periods[0]));
        }
      } catch {
        if (active) {
          setLocalPeriods([]);
        }
      }
    }
    void loadPeriods();
    return () => {
      active = false;
    };
  }, []);

  async function handleRawImport() {
    // Primary path: upload one raw DB1B file and let backend pipeline generate outputs.
    if (!rawFile) {
      return;
    }
    setIsProcessingRaw(true);
    try {
      await onImportRaw(rawFile);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Raw import failed.";
      onImportFailed(message);
    } finally {
      setIsProcessingRaw(false);
    }
  }

  async function handleExistingImport() {
    // Secondary path: import a period that already exists in backend output folders.
    if (!selectedLocalPeriod) {
      return;
    }
    setIsLoadingExisting(true);
    try {
      await onImportExisting(selectedLocalPeriod);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Loading existing period failed.";
      onImportFailed(message);
    } finally {
      setIsLoadingExisting(false);
    }
  }

  return (
    <PageShell
      title="Load Data Set"
      subtitle="Upload one raw DB1B CSV to run parse + analyze, or load existing generated periods"
    >
      <section className="card">
        <header className="card__header">
          <h3 className="card__title">Primary Flow: Raw DB1B Upload</h3>
        </header>
        <div className="card__body">
          <section className="form-grid">
            <label>
              Raw DB1B CSV
              <input
                type="file"
                accept=".csv"
                onChange={(event) => setRawFile(event.target.files?.[0] ?? null)}
              />
            </label>
          </section>

          <div className="page-footer-actions">
            <AppButton
              variant="primary"
              onClick={handleRawImport}
              disabled={!rawFile || isProcessingRaw}
            >
              {isProcessingRaw ? "Running Parse + Analyze..." : "Upload and Process"}
            </AppButton>
          </div>
          {isProcessingRaw ? (
            <p className="load-import-status" aria-live="polite">
              Your File Is Being Processed... {rawElapsedSeconds}s elapsed.
            </p>
          ) : null}
        </div>
      </section>

      <section className="card" style={{ marginTop: "14px" }}>
        <header className="card__header">
          <h3 className="card__title">Load Existing Backend Period</h3>
          <p className="card__subtitle">
            Select a period already present in backend route/hub market power folders.
          </p>
        </header>
        <div className="card__body">
          <section className="form-grid">
            <label>
              Existing Period
              <select
                value={selectedLocalPeriod}
                onChange={(event) => setSelectedLocalPeriod(event.target.value)}
                disabled={localPeriods.length === 0}
              >
                {localPeriods.length === 0 ? <option value="">No periods detected</option> : null}
                {localPeriods.map((period) => (
                  <option key={period} value={period}>
                    {period}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <div className="page-footer-actions">
            <AppButton
              variant="neutral"
              onClick={handleExistingImport}
              disabled={!selectedLocalPeriod || isLoadingExisting}
            >
              {isLoadingExisting ? "Loading Period..." : "Load Selected Period"}
            </AppButton>
          </div>
        </div>
      </section>
    </PageShell>
  );
}






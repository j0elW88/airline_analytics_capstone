/**
 * @file src/app/App.tsx
 * @description Top-level screen router that coordinates navigation, dataset import, and modal rendering.
 */

import { useEffect, useRef } from "react";
import { TopNav } from "../components/layout/TopNav";
import { ModalHost } from "../components/ui/ModalHost";
import { AnalyzeMultiPage } from "../pages/AnalyzeMultiPage";
import { AnalyzeOnePage } from "../pages/AnalyzeOnePage";
import { HistoryPage } from "../pages/HistoryPage";
import { HomePage } from "../pages/HomePage";
import { LoadedDatasetsPage } from "../pages/LoadedDatasetsPage";
import { LoadDatasetPage } from "../pages/LoadDatasetPage";
import { ResultsMultiPage } from "../pages/ResultsMultiPage";
import { ResultsOnePage } from "../pages/ResultsOnePage";
import { StartPage } from "../pages/StartPage";
import { HelpPage } from "../pages/HelpPage";
import { AboutPage } from "../pages/AboutPage";
import { ImportRawError, fetchLocalDataset, fetchLocalPeriods, importRawDb1b } from "../services/localBackend";
import type { HubMarketPowerRow, RouteMarketPowerRow } from "../types/data";
import { getCompletePeriods, getSortedPeriods, useAppState } from "./state";

export function App() {
  // Pull state and action helpers from the global app store.
  const {
    state,
    navTo,
    navBack,
    addHistory,
    upsertDataset,
    setSinglePeriod,
    setMultiPeriods,
    openModal,
    closeModal,
  } = useAppState();

  const sortedPeriods = getSortedPeriods(state);
  const completePeriods = getCompletePeriods(state);

  const selectedDataset = state.selectedSinglePeriod
    ? state.datasetsByPeriod[state.selectedSinglePeriod] ?? null
    : null;

  const selectedMultiDatasets = state.selectedMultiPeriods
    .map((period) => state.datasetsByPeriod[period])
    .filter((dataset): dataset is (typeof state.datasetsByPeriod)[string] => Boolean(dataset));

  // Prevent bootstrap from running more than once in React strict mode double-mount flows.
  const bootRef = useRef(false);

  useEffect(() => {
    if (bootRef.current) {
      return;
    }
    bootRef.current = true;

    async function bootstrapLocalDatasets() {
      try {
        // Ask local dev bridge for periods that already exist in backend output folders.
        const periods = await fetchLocalPeriods();
        for (const period of periods) {
          // Skip periods that are already in in-memory state.
          if (state.datasetsByPeriod[period]) {
            continue;
          }

          try {
            const dataset = await fetchLocalDataset(period);
            upsertDataset({
              period: dataset.period as `${number}_Q${1 | 2 | 3 | 4}`,
              routeRows: dataset.routeRows,
              hubRows: dataset.hubRows,
              uploadedAtIso: new Date().toISOString(),
            });
          } catch {
            // Individual period load errors are intentionally ignored here.
          }
        }
      } catch {
        // Local bridge unavailable (e.g. non-dev host). Manual upload still works.
      }
    }

    void bootstrapLocalDatasets();
  }, []);

  function onImport(payload: {
    period: string;
    routeRows: RouteMarketPowerRow[];
    hubRows: HubMarketPowerRow[];
  }) {
    // Store imported dataset, record activity, then notify user with modal.
    upsertDataset({
      period: payload.period as `${number}_Q${1 | 2 | 3 | 4}`,
      routeRows: payload.routeRows,
      hubRows: payload.hubRows,
      uploadedAtIso: new Date().toISOString(),
    });
    addHistory(`Imported ${payload.period}`);
    openModal({
      title: "Import Complete",
      message: `Loaded ${payload.period} with ${payload.routeRows.length} route rows and ${payload.hubRows.length} hub rows.`,
      actions: [{ label: "Continue", kind: "primary", onClick: closeModal }],
    });
  }

  async function importExistingPeriod(period: string) {
    // Pull route/hub csv for a detected backend period and run standard import flow.
    const dataset = await fetchLocalDataset(period);
    onImport({
      period: dataset.period,
      routeRows: dataset.routeRows,
      hubRows: dataset.hubRows,
    });
  }

  async function importRawFile(file: File) {
    // Send raw DB1B csv to dev bridge, then import the generated analyzed period.
    try {
      const period = await importRawDb1b(file);
      await importExistingPeriod(period);
    } catch (error) {
      if (error instanceof ImportRawError && error.errorType === "verification") {
        throw new Error(`VERIFICATION_FAILED::${error.message}`);
      }
      throw error;
    }
  }

  function renderCurrentScreen() {
    // Single switch keeps navigation deterministic and easy to extend.
    switch (state.screen) {
      case "home":
        return <HomePage onHistory={() => navTo("history")} onLoaded={() => navTo("loaded")} onStart={() => navTo("start")} />;
      case "history":
        return <HistoryPage items={state.history} />;
      case "loaded":
        return <LoadedDatasetsPage periods={sortedPeriods} onAdd={() => navTo("load")} />;
      case "start":
        return (
          <StartPage
            onAnalyzeOne={() => navTo("analyze_one")}
            onAnalyzeMulti={() => navTo("analyze_multi")}
            onLoad={() => navTo("load")}
            onHelp={() => navTo("help")}
            onAbout={() => navTo("about")}
          />
        );
      case "help":
        return <HelpPage onBack={() => navTo("start")} />;
      case "about":
        return <AboutPage />;
      case "load":
        return (
          <LoadDatasetPage
            onImportRaw={importRawFile}
            onImportExisting={importExistingPeriod}
            onImportFailed={(message) => {
              const isVerificationFailure =
                message.startsWith("VERIFICATION_FAILED::") || message.startsWith("VERIFICATION_FAILED");

              const cleanMessage = isVerificationFailure
                ? message.replace(/^VERIFICATION_FAILED::?/, "").trim()
                : message;

              return openModal({
                title: isVerificationFailure ? "Verification Failed" : "Import Failed",
                message: cleanMessage,
                actions: [{ label: "Close", kind: "danger", onClick: closeModal }],
              });
            }}
          />
        );
      case "analyze_one":
        return (
          <AnalyzeOnePage
            periods={completePeriods}
            initialPeriod={state.selectedSinglePeriod}
            onAddDataset={() => navTo("load")}
            onOpenAnalytics={(period) => {
              setSinglePeriod(period);
              addHistory(`Viewed one period: ${period}`);
              navTo("results_one");
            }}
          />
        );
      case "analyze_multi":
        return (
          <AnalyzeMultiPage
            periods={completePeriods}
            initialSelected={state.selectedMultiPeriods}
            onAddDataset={() => navTo("load")}
            onOpenAnalytics={(periods) => {
              setMultiPeriods(periods);
              addHistory(`Viewed multi periods: ${periods.join(", ")}`);
              navTo("results_multi");
            }}
          />
        );
      case "results_one":
        return <ResultsOnePage dataset={selectedDataset} />;
      case "results_multi":
        return <ResultsMultiPage datasets={selectedMultiDatasets} />;
      default:
        return <HomePage onHistory={() => navTo("history")} onLoaded={() => navTo("loaded")} onStart={() => navTo("start")} />;
    }
  }

  return (
    <>
      {/* Home/start hide the top nav for a cleaner launch view. */}
      {state.screen !== "home" && state.screen !== "start" && state.screen !== "help" ? (
        <TopNav showBack onBack={navBack} />
      ) : null}
      {renderCurrentScreen()}
      {/* ModalHost is always mounted so any screen can open an app-level modal. */}
      <ModalHost modal={state.modal} onClose={closeModal} />
    </>
  );
}






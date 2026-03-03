/**
 * @file src/pages/ResultsMultiPage.tsx
 * @description Multi-period results screen combining route and hub insights.
 */

import { useMemo, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { RouteFilterBar } from "../components/filters/RouteFilterBar";
import { Tabs } from "../components/ui/Tabs";
import { EmptyState } from "../components/ui/EmptyState";
import type {
  DatasetRecord,
  HubMarketPowerRow,
  RouteFilters,
  RouteMarketPowerRow,
} from "../types/data";
import { applyHubFilters, applyRouteFilters } from "../features/results/analytics";
import { MarketOverviewPanel } from "../features/results/MarketOverviewPanel";
import { RouteHubInsightsPanel } from "../features/results/RouteHubInsightsPanel";

const routeFilterDefaults: RouteFilters = {
  origin: "",
  dest: "",
  carrier: "",
};

interface ResultsMultiPageProps {
  datasets: DatasetRecord[];
}

function flattenRouteRows(datasets: DatasetRecord[]): RouteMarketPowerRow[] {
  // Merge route rows from selected periods into one analysis array.
  const out: RouteMarketPowerRow[] = [];
  datasets.forEach((dataset) => {
    const routeRows = Array.isArray(dataset.routeRows) ? dataset.routeRows : [];
    routeRows.forEach((row) => {
      out.push({ ...row });
    });
  });
  return out;
}

function flattenHubRows(datasets: DatasetRecord[]): HubMarketPowerRow[] {
  // Merge hub rows from selected periods into one analysis array.
  const out: HubMarketPowerRow[] = [];
  datasets.forEach((dataset) => {
    const hubRows = Array.isArray(dataset.hubRows) ? dataset.hubRows : [];
    hubRows.forEach((row) => {
      out.push({ ...row });
    });
  });
  return out;
}

export function ResultsMultiPage({ datasets }: ResultsMultiPageProps) {
  // Multi-period results shares the same tabs/filters as single-period mode.
  const [activeTab, setActiveTab] = useState<"routexairline" | "hubxairline">("routexairline");
  const [routeFilters, setRouteFilters] = useState<RouteFilters>(routeFilterDefaults);

  const mergedRouteRows = useMemo(() => flattenRouteRows(datasets), [datasets]);
  const mergedHubRows = useMemo(() => flattenHubRows(datasets), [datasets]);

  const filteredRouteRows = useMemo(
    // Apply route filters after period merge.
    () => applyRouteFilters(mergedRouteRows, routeFilters),
    [mergedRouteRows, routeFilters],
  );

  const filteredHubRows = useMemo(
    // Apply carrier-only hub filtering after period merge.
    () => applyHubFilters(mergedHubRows, { carrier: routeFilters.carrier }),
    [mergedHubRows, routeFilters.carrier],
  );

  if (datasets.length === 0) {
    return (
      <PageShell title="Multi-Period Analytics" subtitle="Current phase aggregates loaded periods">
        <EmptyState title="No periods selected" description="Select periods from Analyze Multiple Periods screen." />
      </PageShell>
    );
  }

  const sortedPeriods = datasets.map((dataset) => dataset.period).sort();
  const latestPeriod = sortedPeriods.length > 0 ? sortedPeriods[sortedPeriods.length - 1] : "-";
  const periodLabel = sortedPeriods.join(", ");

  function handleTabChange(key: string) {
    // Reset filters when changing tabs so each view starts from neutral state.
    const next = key === "hubxairline" ? "hubxairline" : "routexairline";
    setActiveTab(next);
    setRouteFilters(routeFilterDefaults);
  }

  return (
    <PageShell
      title="Multi-Period Analytics"
      subtitle={`Combined periods: ${periodLabel}`}
    >
      <Tabs
        options={[
          { key: "routexairline", label: "RouteXAirline" },
          { key: "hubxairline", label: "HubXAirline" },
        ]}
        activeKey={activeTab}
        onChange={handleTabChange}
      />

      {activeTab === "routexairline" ? (
        <>
          <RouteFilterBar
            filters={routeFilters}
            onChange={setRouteFilters}
            period={latestPeriod}
            rows={mergedRouteRows}
          />
          <MarketOverviewPanel rows={filteredRouteRows} filters={routeFilters} />
          <RouteHubInsightsPanel
            routeRows={filteredRouteRows}
            hubRows={filteredHubRows}
            routeFilters={routeFilters}
            view="routexairline"
          />
        </>
      ) : (
        <>
          <RouteFilterBar
            filters={routeFilters}
            onChange={setRouteFilters}
            period={latestPeriod}
            rows={mergedRouteRows}
            showOrigin={false}
            showDestination={false}
          />
          <RouteHubInsightsPanel
            routeRows={filteredRouteRows}
            hubRows={filteredHubRows}
            routeFilters={routeFilters}
            view="hubxairline"
          />
        </>
      )}
    </PageShell>
  );
}






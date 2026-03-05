/**
 * @file src/pages/ResultsOnePage.tsx
 * @description Single-period results screen combining route and hub insights.
 */

import { useMemo, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { RouteFilterBar } from "../components/filters/RouteFilterBar";
import { Tabs } from "../components/ui/Tabs";
import { EmptyState } from "../components/ui/EmptyState";
import type { DatasetRecord, RouteFilters } from "../types/data";
import { applyHubFilters, applyRouteFilters } from "../features/results/analytics";
import { MarketOverviewPanel } from "../features/results/MarketOverviewPanel";
import { RouteHubInsightsPanel } from "../features/results/RouteHubInsightsPanel";

const routeFilterDefaults: RouteFilters = {
  origin: "",
  dest: "",
  carrier: "",
};

interface ResultsOnePageProps {
  dataset: DatasetRecord | null;
}

export function ResultsOnePage({ dataset }: ResultsOnePageProps) {
  // Same two-tab layout used in multi-period view for consistent UX.
  const [activeTab, setActiveTab] = useState<"routexairline" | "hubxairline">("routexairline");
  const [routeFilters, setRouteFilters] = useState<RouteFilters>(routeFilterDefaults);
  const routeRows = Array.isArray(dataset?.routeRows) ? dataset.routeRows : [];
  const hubRows = Array.isArray(dataset?.hubRows) ? dataset.hubRows : [];

  const filteredRouteRows = useMemo(
    // Route-side filters apply to route rows only.
    () => applyRouteFilters(routeRows, routeFilters),
    [routeRows, routeFilters],
  );

  const filteredHubRows = useMemo(
    // Hub view only uses carrier filter from the shared route filter bar.
    () => applyHubFilters(hubRows, { carrier: routeFilters.carrier }),
    [hubRows, routeFilters.carrier],
  );

  if (!dataset) {
    return (
      <PageShell title="Analytics" subtitle="Single period results">
        <EmptyState title="No period selected" description="Select a period from Analyze One screen first." />
      </PageShell>
    );
  }

  function handleTabChange(key: string) {
    // Clear filters when switching tabs to avoid stale context carry-over.
    const next = key === "hubxairline" ? "hubxairline" : "routexairline";
    setActiveTab(next);
    setRouteFilters(routeFilterDefaults);
  }

  return (
    <PageShell
      title={`Analytics: ${dataset.period}`}
      subtitle="Single period analysis"
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
            period={dataset.period}
            rows={routeRows}
          />
          <MarketOverviewPanel rows={filteredRouteRows} filters={routeFilters} />
          <RouteHubInsightsPanel
            period={dataset.period}
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
            period={dataset.period}
            rows={routeRows}
            showOrigin={false}
            showDestination={false}
          />
          <RouteHubInsightsPanel
            period={dataset.period}
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






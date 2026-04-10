/**
 * @file src/pages/ResultsMultiPage.tsx
 * @description Multi-period results page with full per-period analytics and advanced comparative analysis.
 */

import { useEffect, useMemo, useState } from "react";
import { LineTrendChart } from "../components/charts/LineTrendChart";
import { SimpleBarChart } from "../components/charts/SimpleBarChart";
import { RouteFilterBar } from "../components/filters/RouteFilterBar";
import { useCarrierLookup } from "../hooks/useCarrierLookup";
import { PageShell } from "../components/layout/PageShell";
import { Card } from "../components/ui/Card";
import { DataTable, type DataColumn } from "../components/ui/DataTable";
import { EmptyState } from "../components/ui/EmptyState";
import { MetricCard } from "../components/ui/MetricCard";
import { Tabs } from "../components/ui/Tabs";
import { AppButton } from "../components/ui/AppButton";
import { getCarrierDisplayName, normalizeCarrierCode } from "../utils/carrierDisplay";
import { applyHubFilters, applyRouteFilters } from "../features/results/analytics";
import {
  buildFareFrequencyBandShift,
  buildHubComparativeSeries,
  buildHubPassengerChange,
  buildPeriodValueChangeRows,
  buildRouteCarrierShareShift,
  buildRouteComparativeSeries,
  buildRouteFareFrequencyBands,
  buildRouteMarketFareChange,
  computeTrendDelta,
  pickRoutePrice,
  sortDatasetsByPeriod,
  toTrendPoints,
  type ComparativeChangeRow,
  type FareFrequencyBandSeries,
  type RoutePriceMetric,
  type TrendDelta,
} from "../features/results/comparative";
import { MarketOverviewPanel } from "../features/results/MarketOverviewPanel";
import { RouteHubInsightsPanel } from "../features/results/RouteHubInsightsPanel";
import type {
  DatasetRecord,
  HubMarketPowerRow,
  RouteFilters,
  RouteMarketPowerRow,
} from "../types/data";
import { formatRouteDisplay, getAirportDisplayName, normalizeAirportCode } from "../utils/airports";
import { formatCurrency, formatNumber } from "../utils/format";
import { parseLocationSelection } from "../utils/locationTaxonomy";
import {
  fetchHubFareDistribution,
  fetchRouteFareDistribution,
  type FareDistributionBin,
  type HubFareDistributionResponse,
  type RouteFareDistributionResponse,
} from "../services/localBackend";

const COMPARATIVE_TAB_KEY = "__comparative__";
const MIN_COMPARE_PERIODS = 2;
const MIN_ROUTE_FARE_CHANGE_PASSENGERS = 5000;
const MIN_HUB_PASSENGER_CHANGE_PASSENGERS = 5000;

const routeFilterDefaults: RouteFilters = {
  origin: "",
  dest: "",
  carrier: "",
};

const routePriceMetricLabels: Record<RoutePriceMetric, string> = {
  avg: "Avg Fare",
  max: "Max Fare",
  min: "Min Fare",
  median: "Median Fare",
};

const priceMetricOptions: Array<{ value: RoutePriceMetric; label: string }> = [
  { value: "avg", label: "Average Price" },
  { value: "max", label: "Max Price" },
  { value: "min", label: "Min Price" },
  { value: "median", label: "Median Price" },
];

const trendPalette = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "#2f855a",
  "#b7791f",
  "#1f4e79",
];

const hubCarrierActivityColumns: DataColumn<HubCarrierActivityRow>[] = [
  { key: "carrier", header: "Carrier", render: (row) => row.carrierLabel },
  { key: "passengers", header: "Passengers", render: (row) => formatNumber(row.passengers) },
  { key: "share", header: "Hub Share", render: (row) => `${row.sharePct.toFixed(1)}%` },
  { key: "destinations", header: "Destinations", render: (row) => formatNumber(row.destinationsServed) },
  { key: "topRoute", header: "Top Carrier Route", render: (row) => row.topRoute },
  { key: "topRoutePassengers", header: "Top Route Pax", render: (row) => formatNumber(row.topRoutePassengers) },
];

interface ResultsMultiPageProps {
  datasets: DatasetRecord[];
}

function flattenRouteRows(datasets: DatasetRecord[]): RouteMarketPowerRow[] {
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
  const out: HubMarketPowerRow[] = [];
  datasets.forEach((dataset) => {
    const hubRows = Array.isArray(dataset.hubRows) ? dataset.hubRows : [];
    hubRows.forEach((row) => {
      out.push({ ...row });
    });
  });
  return out;
}

function formatDeltaHint(
  delta: TrendDelta,
  valueFormatter: (value: number) => string,
): string {
  const changePrefix = delta.startLabel ? `Vs ${delta.startLabel}` : "Change";
  const sign = delta.absolute > 0 ? "+" : "";
  const absoluteText = `${sign}${valueFormatter(delta.absolute)}`;
  if (delta.pct === null) {
    return `${changePrefix}: ${absoluteText}`;
  }
  const pctSign = delta.pct > 0 ? "+" : "";
  return `${changePrefix}: ${absoluteText} (${pctSign}${delta.pct.toFixed(1)}%)`;
}

function toBarRows(rows: ComparativeChangeRow[]): Array<{ label: string; value: number }> {
  return rows.map((row) => ({
    label: row.label,
    value: row.value,
  }));
}

function getSharedBarMax(groups: Array<Array<{ value: number }>>): number {
  let maxValue = 0;
  groups.forEach((rows) => {
    rows.forEach((row) => {
      if (Number.isFinite(row.value) && row.value > maxValue) {
        maxValue = row.value;
      }
    });
  });
  return maxValue;
}

function formatDeltaCell(delta: TrendDelta, valueFormatter: (value: number) => string): string {
  const sign = delta.absolute > 0 ? "+" : "";
  const absolute = `${sign}${valueFormatter(delta.absolute)}`;
  if (delta.pct === null) {
    return absolute;
  }
  const pctSign = delta.pct > 0 ? "+" : "";
  return `${absolute} (${pctSign}${delta.pct.toFixed(1)}%)`;
}

interface ContributionStats {
  rows: number;
  markets: number;
  passengers: number;
}

interface TopCarrierContributionRow {
  carrier: string;
  sharePct: number;
  passengers: number;
}

interface TopCarrierShareDetail {
  period: string;
  topSharePct: number;
  totalPassengers: number;
  topCarriers: TopCarrierContributionRow[];
}

interface HubCarrierActivityRow {
  carrierCode: string;
  carrierLabel: string;
  passengers: number;
  sharePct: number;
  destinationsServed: number;
  topRoute: string;
  topRoutePassengers: number;
}

interface HubFareStatsPoint {
  period: string;
  avgFare: number;
  medianFare: number;
  iqrFare: number;
  lowFare: number;
  highFare: number;
  topContributingCarriers: string[];
  lowFareCarriers: string[];
  highFareCarriers: string[];
}

interface CarrierFareTrendSeries {
  carrier: string;
  carrierName?: string;
  totalPassengers: number;
  periodsWithData: number;
  avgFareTrend: Array<{ label: string; value: number }>;
  medianFareTrend: Array<{ label: string; value: number }>;
  iqrTrend: Array<{ label: string; value: number }>;
  passengersTrend: Array<{ label: string; value: number }>;
  shareTrend: Array<{ label: string; value: number }>;
}

interface RouteCarrierDistributionState {
  status: "idle" | "loading" | "ready" | "error";
  key: string;
  responses: RouteFareDistributionResponse[];
  missingPeriods: string[];
  error: string;
}

interface HubFareDistributionState {
  status: "idle" | "loading" | "ready" | "error";
  key: string;
  responses: HubFareDistributionResponse[];
  missingPeriods: string[];
  error: string;
}

function sortFareBins(bins: FareDistributionBin[]): FareDistributionBin[] {
  return [...bins].sort((a, b) => a.fareStart - b.fareStart);
}

function computeAverageFromFareBins(bins: FareDistributionBin[]): number {
  const sorted = sortFareBins(bins);
  let weightedSum = 0;
  let totalPassengers = 0;
  sorted.forEach((bin) => {
    const passengers = Number(bin.passengers ?? 0);
    const start = Number(bin.fareStart ?? 0);
    const end = Number(bin.fareEnd ?? 0);
    if (!Number.isFinite(passengers) || passengers <= 0 || !Number.isFinite(start) || !Number.isFinite(end)) {
      return;
    }
    const midpoint = (start + end) / 2;
    weightedSum += midpoint * passengers;
    totalPassengers += passengers;
  });
  return totalPassengers > 0 ? weightedSum / totalPassengers : 0;
}

function computeQuantileFromFareBins(bins: FareDistributionBin[], quantile: number): number {
  const sorted = sortFareBins(bins);
  const totalPassengers = sorted.reduce((sum, bin) => {
    const passengers = Number(bin.passengers ?? 0);
    return Number.isFinite(passengers) && passengers > 0 ? sum + passengers : sum;
  }, 0);
  if (!(totalPassengers > 0)) {
    return 0;
  }
  const boundedQuantile = Math.max(0, Math.min(1, quantile));
  const target = totalPassengers * boundedQuantile;
  let running = 0;
  for (const bin of sorted) {
    const passengers = Number(bin.passengers ?? 0);
    const start = Number(bin.fareStart ?? 0);
    const end = Number(bin.fareEnd ?? start);
    if (!Number.isFinite(passengers) || passengers <= 0 || !Number.isFinite(start) || !Number.isFinite(end)) {
      continue;
    }
    const next = running + passengers;
    if (target <= next) {
      const fraction = passengers > 0 ? (target - running) / passengers : 0;
      if (end > start) {
        return start + (Math.max(0, Math.min(1, fraction)) * (end - start));
      }
      return start;
    }
    running = next;
  }
  const last = sorted[sorted.length - 1];
  return Number(last?.fareEnd ?? last?.fareStart ?? 0);
}

function buildCarrierFareTrendsFromDistributions(
  periodOrder: string[],
  responses: RouteFareDistributionResponse[],
): CarrierFareTrendSeries[] {
  const responseByPeriod = new Map(responses.map((response) => [response.period, response]));
  const routePassengersByPeriod = new Map<string, number>();
  responses.forEach((response) => {
    const total = response.carriers.reduce((sum, carrier) => {
      const passengers = Number(carrier.totalPassengers ?? 0);
      return Number.isFinite(passengers) && passengers > 0 ? sum + passengers : sum;
    }, 0);
    routePassengersByPeriod.set(response.period, total);
  });
  const carrierSet = new Set<string>();
  responses.forEach((response) => {
    response.carriers.forEach((carrier) => {
      const code = normalizeCarrierCode(carrier.carrier);
      if (code) {
        carrierSet.add(code);
      }
    });
  });

  const out: CarrierFareTrendSeries[] = [];
  carrierSet.forEach((carrierCode) => {
    const avgFareTrend: Array<{ label: string; value: number }> = [];
    const medianFareTrend: Array<{ label: string; value: number }> = [];
    const iqrTrend: Array<{ label: string; value: number }> = [];
    const passengersTrend: Array<{ label: string; value: number }> = [];
    const shareTrend: Array<{ label: string; value: number }> = [];
    let periodsWithData = 0;
    let totalPassengers = 0;
    let carrierName = "";

    periodOrder.forEach((period) => {
      const response = responseByPeriod.get(period);
      const carrier = response?.carriers.find((item) => normalizeCarrierCode(item.carrier) === carrierCode);
      if (!carrier || !Array.isArray(carrier.bins) || carrier.bins.length === 0 || !(carrier.totalPassengers > 0)) {
        return;
      }
      const avg = computeAverageFromFareBins(carrier.bins);
      const q1 = computeQuantileFromFareBins(carrier.bins, 0.25);
      const median = computeQuantileFromFareBins(carrier.bins, 0.5);
      const q3 = computeQuantileFromFareBins(carrier.bins, 0.75);
      const iqr = Math.max(q3 - q1, 0);
      const passengers = Number(carrier.totalPassengers ?? 0);
      const routePassengers = Number(routePassengersByPeriod.get(period) ?? 0);
      const sharePct = routePassengers > 0 ? (passengers / routePassengers) * 100 : 0;
      avgFareTrend.push({ label: period, value: avg });
      medianFareTrend.push({ label: period, value: median });
      iqrTrend.push({ label: period, value: iqr });
      passengersTrend.push({ label: period, value: passengers });
      shareTrend.push({ label: period, value: sharePct });
      periodsWithData += 1;
      totalPassengers += passengers;
      if (!carrierName) {
        carrierName = String(carrier.carrierName ?? "").trim();
      }
    });

    if (periodsWithData === 0) {
      return;
    }
    out.push({
      carrier: carrierCode,
      carrierName,
      totalPassengers,
      periodsWithData,
      avgFareTrend,
      medianFareTrend,
      iqrTrend,
      passengersTrend,
      shareTrend,
    });
  });

  return out.sort((a, b) => b.totalPassengers - a.totalPassengers);
}

function buildMetricTooltip(
  title: string,
  definition: string,
  delta: TrendDelta,
  valueFormatter: (value: number) => string,
  contributionLines: string[] = [],
): string {
  const baseLabel = delta.startLabel ?? "Base";
  const compareLabel = delta.endLabel ?? "Comparison";
  const baseLine = `${baseLabel}: ${valueFormatter(delta.start)}`;
  const compareLine = `${compareLabel}: ${valueFormatter(delta.end)}`;
  const sign = delta.absolute > 0 ? "+" : "";
  const absoluteText = `${sign}${valueFormatter(delta.absolute)}`;
  const changeLine = delta.pct === null
    ? `Change: ${absoluteText}`
    : `Change: ${absoluteText} (${delta.pct > 0 ? "+" : ""}${delta.pct.toFixed(1)}%)`;
  return [
    title,
    definition,
    baseLine,
    compareLine,
    changeLine,
    ...contributionLines,
  ].join("\n");
}

function buildTopCarrierShareDetailsForRoute(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
): TopCarrierShareDetail[] {
  return datasets.map((dataset) => {
    const rows = applyRouteFilters(Array.isArray(dataset.routeRows) ? dataset.routeRows : [], routeFilters);
    const byCarrier = new Map<string, number>();
    let totalPassengers = 0;
    rows.forEach((row) => {
      const carrier = normalizeCarrierCode(String(row.Carrier ?? ""));
      const passengers = Number(row.total_passengers ?? 0);
      if (!carrier || !Number.isFinite(passengers) || passengers <= 0) {
        return;
      }
      byCarrier.set(carrier, (byCarrier.get(carrier) ?? 0) + passengers);
      totalPassengers += passengers;
    });
    if (!(totalPassengers > 0) || byCarrier.size === 0) {
      return {
        period: dataset.period,
        topSharePct: 0,
        totalPassengers: 0,
        topCarriers: [],
      };
    }
    const entries = Array.from(byCarrier.entries())
      .map(([carrier, passengers]) => ({
        carrier,
        passengers,
        sharePct: (passengers / totalPassengers) * 100,
      }))
      .sort((a, b) => b.passengers - a.passengers);
    const topPassengers = entries[0]?.passengers ?? 0;
    const topCarriers = entries.filter((entry) => Math.abs(entry.passengers - topPassengers) < 1e-6);
    return {
      period: dataset.period,
      topSharePct: entries[0]?.sharePct ?? 0,
      totalPassengers,
      topCarriers,
    };
  });
}

function buildTopCarrierShareDetailsForHub(
  datasets: DatasetRecord[],
  carrierFilter: string,
  hubOrigin: string,
): TopCarrierShareDetail[] {
  const hubCode = hubOrigin.trim().toUpperCase();
  return datasets.map((dataset) => {
    const scopedRows = applyHubFilters(Array.isArray(dataset.hubRows) ? dataset.hubRows : [], { carrier: carrierFilter });
    const rows = hubCode
      ? scopedRows.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === hubCode)
      : scopedRows;
    const byCarrier = new Map<string, number>();
    let totalPassengers = 0;
    rows.forEach((row) => {
      const carrier = normalizeCarrierCode(String(row.Carrier ?? ""));
      const passengers = Number(row.total_passengers ?? 0);
      if (!carrier || !Number.isFinite(passengers) || passengers <= 0) {
        return;
      }
      byCarrier.set(carrier, (byCarrier.get(carrier) ?? 0) + passengers);
      totalPassengers += passengers;
    });
    if (!(totalPassengers > 0) || byCarrier.size === 0) {
      return {
        period: dataset.period,
        topSharePct: 0,
        totalPassengers: 0,
        topCarriers: [],
      };
    }
    const entries = Array.from(byCarrier.entries())
      .map(([carrier, passengers]) => ({
        carrier,
        passengers,
        sharePct: (passengers / totalPassengers) * 100,
      }))
      .sort((a, b) => b.passengers - a.passengers);
    const topPassengers = entries[0]?.passengers ?? 0;
    const topCarriers = entries.filter((entry) => Math.abs(entry.passengers - topPassengers) < 1e-6);
    return {
      period: dataset.period,
      topSharePct: entries[0]?.sharePct ?? 0,
      totalPassengers,
      topCarriers,
    };
  });
}

function buildHubFareStatsByPeriod(
  periodOrder: string[],
  responses: HubFareDistributionResponse[],
): HubFareStatsPoint[] {
  const byPeriod = new Map(responses.map((response) => [response.period, response]));
  const out: HubFareStatsPoint[] = [];

  periodOrder.forEach((period) => {
    const response = byPeriod.get(period);
    if (!response || !Array.isArray(response.carriers) || response.carriers.length === 0) {
      return;
    }
    const allBins: FareDistributionBin[] = [];
    let lowFare = Number.POSITIVE_INFINITY;
    let highFare = Number.NEGATIVE_INFINITY;
    const lowFareCarriers = new Set<string>();
    const highFareCarriers = new Set<string>();

    response.carriers.forEach((carrier) => {
      const carrierCode = normalizeCarrierCode(String(carrier.carrier ?? ""));
      const carrierBins = Array.isArray(carrier.bins)
        ? carrier.bins.filter((bin) => {
          const start = Number(bin.fareStart ?? 0);
          const end = Number(bin.fareEnd ?? 0);
          const passengers = Number(bin.passengers ?? 0);
          return Number.isFinite(start) && Number.isFinite(end) && end > start && Number.isFinite(passengers) && passengers > 0;
        })
        : [];
      if (!carrierCode || carrierBins.length === 0) {
        return;
      }
      allBins.push(...carrierBins);
      const minFromBins = Math.min(...carrierBins.map((bin) => Number(bin.fareStart ?? Number.POSITIVE_INFINITY)));
      const maxFromBins = Math.max(...carrierBins.map((bin) => Number(bin.fareEnd ?? Number.NEGATIVE_INFINITY)));
      const carrierMinFare = Number.isFinite(Number(carrier.minFare)) && Number(carrier.minFare) > 0
        ? Number(carrier.minFare)
        : minFromBins;
      const carrierMaxFare = Number.isFinite(Number(carrier.maxFare)) && Number(carrier.maxFare) > 0
        ? Number(carrier.maxFare)
        : maxFromBins;

      if (carrierMinFare < lowFare) {
        lowFare = carrierMinFare;
        lowFareCarriers.clear();
        lowFareCarriers.add(carrierCode);
      } else if (Math.abs(carrierMinFare - lowFare) < 1e-6) {
        lowFareCarriers.add(carrierCode);
      }

      if (carrierMaxFare > highFare) {
        highFare = carrierMaxFare;
        highFareCarriers.clear();
        highFareCarriers.add(carrierCode);
      } else if (Math.abs(carrierMaxFare - highFare) < 1e-6) {
        highFareCarriers.add(carrierCode);
      }
    });

    if (allBins.length === 0) {
      return;
    }

    const avgFare = computeAverageFromFareBins(allBins);
    const medianFare = computeQuantileFromFareBins(allBins, 0.5);
    const q1 = computeQuantileFromFareBins(allBins, 0.25);
    const q3 = computeQuantileFromFareBins(allBins, 0.75);
    const topContributingCarriers = response.carriers
      .filter((carrier) => Number(carrier.totalPassengers ?? 0) > 0)
      .slice(0, 3)
      .map((carrier) => normalizeCarrierCode(String(carrier.carrier ?? "")))
      .filter(Boolean);
    out.push({
      period,
      avgFare,
      medianFare,
      iqrFare: Math.max(q3 - q1, 0),
      lowFare: Number.isFinite(lowFare) ? lowFare : 0,
      highFare: Number.isFinite(highFare) ? highFare : 0,
      topContributingCarriers,
      lowFareCarriers: Array.from(lowFareCarriers),
      highFareCarriers: Array.from(highFareCarriers),
    });
  });
  return out;
}


export function ResultsMultiPage({ datasets }: ResultsMultiPageProps) {
  const carrierLookup = useCarrierLookup();
  const sortedDatasets = useMemo(() => sortDatasetsByPeriod(datasets), [datasets]);
  const allRouteRows = useMemo(() => flattenRouteRows(sortedDatasets), [sortedDatasets]);
  const allHubRows = useMemo(() => flattenHubRows(sortedDatasets), [sortedDatasets]);
  const sortedPeriodKeys = useMemo(() => sortedDatasets.map((dataset) => String(dataset.period)), [sortedDatasets]);

  const [activeView, setActiveView] = useState<"routexairline" | "hubxairline">("routexairline");
  const [activePeriodTab, setActivePeriodTab] = useState<string>(() => sortedDatasets[0]?.period ?? COMPARATIVE_TAB_KEY);
  const [routeFilters, setRouteFilters] = useState<RouteFilters>(routeFilterDefaults);
  const [routePriceMetric, setRoutePriceMetric] = useState<RoutePriceMetric>("avg");
  const [hubOrigin, setHubOrigin] = useState("");
  const [comparisonPeriods, setComparisonPeriods] = useState<string[]>(() => sortedDatasets.map((dataset) => dataset.period));
  const [baseComparisonPeriod, setBaseComparisonPeriod] = useState<string>(() => sortedDatasets[0]?.period ?? "");
  const [showAllRouteCarriers, setShowAllRouteCarriers] = useState(false);
  const [routeCarrierDistributionState, setRouteCarrierDistributionState] = useState<RouteCarrierDistributionState>({
    status: "idle",
    key: "",
    responses: [],
    missingPeriods: [],
    error: "",
  });
  const [hubFareDistributionState, setHubFareDistributionState] = useState<HubFareDistributionState>({
    status: "idle",
    key: "",
    responses: [],
    missingPeriods: [],
    error: "",
  });

  const periodTabs = useMemo(
    () => [
      ...sortedDatasets.map((dataset) => ({ key: dataset.period, label: dataset.period })),
      { key: COMPARATIVE_TAB_KEY, label: "Comparative" },
    ],
    [sortedDatasets],
  );

  useEffect(() => {
    const validKeys = new Set(periodTabs.map((tab) => tab.key));
    if (!validKeys.has(activePeriodTab)) {
      setActivePeriodTab(sortedDatasets[0]?.period ?? COMPARATIVE_TAB_KEY);
    }
  }, [activePeriodTab, periodTabs, sortedDatasets]);

  useEffect(() => {
    setComparisonPeriods((previous) => {
      const valid = previous.filter((period) => sortedPeriodKeys.includes(period));
      let next = valid.length > 0 ? valid : sortedPeriodKeys.slice(0, MIN_COMPARE_PERIODS);
      for (const period of sortedPeriodKeys) {
        if (next.length >= MIN_COMPARE_PERIODS) {
          break;
        }
        if (!next.includes(period)) {
          next = [...next, period];
        }
      }
      const ordered = sortedPeriodKeys.filter((period) => next.includes(period));
      if (ordered.length === previous.length && ordered.every((period, index) => period === previous[index])) {
        return previous;
      }
      return ordered;
    });
  }, [sortedPeriodKeys]);

  useEffect(() => {
    setBaseComparisonPeriod((previous) => {
      if (comparisonPeriods.includes(previous)) {
        return previous;
      }
      return comparisonPeriods[0] ?? sortedPeriodKeys[0] ?? "";
    });
  }, [comparisonPeriods, sortedPeriodKeys]);

  useEffect(() => {
    setShowAllRouteCarriers(false);
  }, [routeFilters.origin, routeFilters.dest, activeView, activePeriodTab]);

  if (sortedDatasets.length < 2) {
    return (
      <PageShell title="Multi-Period Analytics" subtitle="Comparative mode requires at least two periods">
        <EmptyState
          title="Need at least 2 selected periods"
          description="Go back to Analyze Multiple Periods and choose two to five complete periods."
        />
      </PageShell>
    );
  }

  const comparisonPeriodSet = new Set(comparisonPeriods);
  const comparisonDatasets = sortedDatasets.filter((dataset) => comparisonPeriodSet.has(dataset.period));
  const orderedComparisonPeriods = sortedPeriodKeys.filter((period) => comparisonPeriodSet.has(period));
  const orderedComparisonPeriodsKey = orderedComparisonPeriods.join(",");
  const hasEnoughComparisonPeriods = comparisonDatasets.length >= MIN_COMPARE_PERIODS;
  const effectiveBasePeriod = comparisonPeriods.includes(baseComparisonPeriod)
    ? baseComparisonPeriod
    : (comparisonPeriods[0] ?? sortedPeriodKeys[0] ?? "");

  const selectedDataset = sortedDatasets.find((dataset) => dataset.period === activePeriodTab) ?? null;
  const filterPeriodLabel = selectedDataset?.period ?? sortedDatasets[sortedDatasets.length - 1]?.period ?? "-";

  const selectedRouteRows = selectedDataset
    ? applyRouteFilters(selectedDataset.routeRows, routeFilters)
    : [];
  const selectedRouteRowsForHub = selectedDataset
    ? applyRouteFilters(selectedDataset.routeRows, {
      origin: hubOrigin,
      dest: "",
      carrier: routeFilters.carrier,
    })
    : [];
  const selectedHubRows = selectedDataset
    ? applyHubFilters(selectedDataset.hubRows, { carrier: routeFilters.carrier }).filter((row) => (
      !hubOrigin || String(row.Origin ?? "").trim().toUpperCase() === hubOrigin.trim().toUpperCase()
    ))
    : [];

  const originSelection = parseLocationSelection(routeFilters.origin);
  const destinationSelection = parseLocationSelection(routeFilters.dest);
  const isSpecificRouteSelection = originSelection.type === "airport" && destinationSelection.type === "airport";
  const selectedRouteLabel = isSpecificRouteSelection
    ? formatRouteDisplay(originSelection.code, destinationSelection.code)
    : "";
  const selectedHubCode = hubOrigin.trim().toUpperCase();
  const isSpecificHubSelection = selectedHubCode.length === 3;
  const selectedHubLabel = isSpecificHubSelection ? getAirportDisplayName(selectedHubCode) : "All Hubs";

  useEffect(() => {
    const orderedPeriods = orderedComparisonPeriodsKey ? orderedComparisonPeriodsKey.split(",") : [];
    const canLoadRouteCarrierDistributions = activeView === "routexairline"
      && activePeriodTab === COMPARATIVE_TAB_KEY
      && hasEnoughComparisonPeriods
      && isSpecificRouteSelection
      && orderedPeriods.length >= MIN_COMPARE_PERIODS;
    if (!canLoadRouteCarrierDistributions) {
      setRouteCarrierDistributionState((previous) => {
        if (previous.status === "idle" && previous.key === "") {
          return previous;
        }
        return {
          status: "idle",
          key: "",
          responses: [],
          missingPeriods: [],
          error: "",
        };
      });
      return;
    }

    const scopeKey = `${orderedComparisonPeriodsKey}|${originSelection.code}|${destinationSelection.code}`;
    let cancelled = false;
    setRouteCarrierDistributionState((previous) => {
      if (previous.key === scopeKey && (previous.status === "loading" || previous.status === "ready")) {
        return previous;
      }
      return {
        status: "loading",
        key: scopeKey,
        responses: [],
        missingPeriods: [],
        error: "",
      };
    });

    void Promise.allSettled(
      orderedPeriods.map((period) => fetchRouteFareDistribution({
        period,
        origin: originSelection.code,
        dest: destinationSelection.code,
      })),
    ).then((results) => {
      if (cancelled) {
        return;
      }
      const successes: RouteFareDistributionResponse[] = [];
      const missingPeriods: string[] = [];
      const errors: string[] = [];
      results.forEach((result, index) => {
        const period = orderedPeriods[index];
        if (result.status === "fulfilled") {
          successes.push(result.value);
          return;
        }
        missingPeriods.push(period);
        const message = result.reason instanceof Error ? result.reason.message : String(result.reason ?? "");
        if (message) {
          errors.push(`${period}: ${message}`);
        }
      });

      if (successes.length === 0) {
        setRouteCarrierDistributionState({
          status: "error",
          key: scopeKey,
          responses: [],
          missingPeriods,
          error: errors[0] ?? "No fare-distribution data found for selected periods and route.",
        });
        return;
      }

      const responseByPeriod = new Map(successes.map((response) => [response.period, response]));
      const orderedResponses = orderedPeriods
        .map((period) => responseByPeriod.get(period))
        .filter((response): response is RouteFareDistributionResponse => Boolean(response));
      setRouteCarrierDistributionState({
        status: "ready",
        key: scopeKey,
        responses: orderedResponses,
        missingPeriods,
        error: errors.join(" | "),
      });
    }).catch((error: unknown) => {
      if (cancelled) {
        return;
      }
      const message = error instanceof Error ? error.message : "Failed loading route fare distributions.";
      setRouteCarrierDistributionState({
        status: "error",
        key: scopeKey,
        responses: [],
        missingPeriods: [...orderedPeriods],
        error: message,
      });
    });

    return () => {
      cancelled = true;
    };
  }, [
    activePeriodTab,
    activeView,
    destinationSelection.code,
    hasEnoughComparisonPeriods,
    isSpecificRouteSelection,
    orderedComparisonPeriodsKey,
    originSelection.code,
  ]);

  useEffect(() => {
    const orderedPeriods = orderedComparisonPeriodsKey ? orderedComparisonPeriodsKey.split(",") : [];
    const normalizedCarrier = normalizeCarrierCode(routeFilters.carrier);
    const canLoadHubFareDistributions = activeView === "hubxairline"
      && activePeriodTab === COMPARATIVE_TAB_KEY
      && hasEnoughComparisonPeriods
      && orderedPeriods.length >= MIN_COMPARE_PERIODS;
    if (!canLoadHubFareDistributions) {
      setHubFareDistributionState((previous) => {
        if (previous.status === "idle" && previous.key === "") {
          return previous;
        }
        return {
          status: "idle",
          key: "",
          responses: [],
          missingPeriods: [],
          error: "",
        };
      });
      return;
    }

    const scopeKey = `${orderedComparisonPeriodsKey}|${selectedHubCode || "ALL"}|${normalizedCarrier}`;
    let cancelled = false;
    setHubFareDistributionState((previous) => {
      if (previous.key === scopeKey && (previous.status === "loading" || previous.status === "ready")) {
        return previous;
      }
      return {
        status: "loading",
        key: scopeKey,
        responses: [],
        missingPeriods: [],
        error: "",
      };
    });

    void Promise.allSettled(
      orderedPeriods.map((period) => fetchHubFareDistribution({
        period,
        origin: selectedHubCode || undefined,
        carrier: normalizedCarrier || undefined,
      })),
    ).then((results) => {
      if (cancelled) {
        return;
      }
      const successes: HubFareDistributionResponse[] = [];
      const missingPeriods: string[] = [];
      const errors: string[] = [];
      results.forEach((result, index) => {
        const period = orderedPeriods[index];
        if (result.status === "fulfilled") {
          successes.push(result.value);
          return;
        }
        missingPeriods.push(period);
        const message = result.reason instanceof Error ? result.reason.message : String(result.reason ?? "");
        if (message) {
          errors.push(`${period}: ${message}`);
        }
      });

      if (successes.length === 0) {
        setHubFareDistributionState({
          status: "error",
          key: scopeKey,
          responses: [],
          missingPeriods,
          error: errors[0] ?? "No DB1B hub fare-distribution data found for selected periods.",
        });
        return;
      }

      const byPeriod = new Map(successes.map((response) => [response.period, response]));
      const orderedResponses = orderedPeriods
        .map((period) => byPeriod.get(period))
        .filter((response): response is HubFareDistributionResponse => Boolean(response));
      setHubFareDistributionState({
        status: "ready",
        key: scopeKey,
        responses: orderedResponses,
        missingPeriods,
        error: errors.join(" | "),
      });
    }).catch((error: unknown) => {
      if (cancelled) {
        return;
      }
      const message = error instanceof Error ? error.message : "Failed loading hub fare distributions.";
      setHubFareDistributionState({
        status: "error",
        key: scopeKey,
        responses: [],
        missingPeriods: [...orderedPeriods],
        error: message,
      });
    });

    return () => {
      cancelled = true;
    };
  }, [
    activePeriodTab,
    activeView,
    hasEnoughComparisonPeriods,
    orderedComparisonPeriodsKey,
    routeFilters.carrier,
    selectedHubCode,
  ]);

  const routeContributionByPeriod = useMemo(() => {
    const map = new Map<string, ContributionStats>();
    comparisonDatasets.forEach((dataset) => {
      const rows = applyRouteFilters(Array.isArray(dataset.routeRows) ? dataset.routeRows : [], routeFilters);
      const routeSet = new Set<string>();
      let passengers = 0;
      rows.forEach((row) => {
        routeSet.add(`${String(row.Origin ?? "").trim().toUpperCase()}-${String(row.Dest ?? "").trim().toUpperCase()}`);
        passengers += Number(row.total_passengers ?? 0);
      });
      map.set(dataset.period, {
        rows: rows.length,
        markets: routeSet.size,
        passengers,
      });
    });
    return map;
  }, [comparisonDatasets, routeFilters]);

  const hubContributionByPeriod = useMemo(() => {
    const map = new Map<string, ContributionStats>();
    comparisonDatasets.forEach((dataset) => {
      const scopedRows = applyHubFilters(Array.isArray(dataset.hubRows) ? dataset.hubRows : [], { carrier: routeFilters.carrier });
      const rows = hubOrigin
        ? scopedRows.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === hubOrigin.trim().toUpperCase())
        : scopedRows;
      const hubSet = new Set<string>();
      let passengers = 0;
      rows.forEach((row) => {
        hubSet.add(normalizeAirportCode(row.Origin));
        passengers += Number(row.total_passengers ?? 0);
      });
      map.set(dataset.period, {
        rows: rows.length,
        markets: Array.from(hubSet).filter(Boolean).length,
        passengers,
      });
    });
    return map;
  }, [comparisonDatasets, hubOrigin, routeFilters.carrier]);

  const routeComparative = useMemo(
    () => buildRouteComparativeSeries(comparisonDatasets, routeFilters),
    [comparisonDatasets, routeFilters],
  );
  const routeTopCarrierShareDetails = useMemo(
    () => buildTopCarrierShareDetailsForRoute(comparisonDatasets, routeFilters),
    [comparisonDatasets, routeFilters],
  );
  const routeAllCarrierFareTrends = useMemo(
    () => {
      if (!isSpecificRouteSelection || routeCarrierDistributionState.status !== "ready") {
        return [];
      }
      const periods = orderedComparisonPeriodsKey ? orderedComparisonPeriodsKey.split(",") : [];
      if (periods.length === 0) {
        return [];
      }
      return buildCarrierFareTrendsFromDistributions(periods, routeCarrierDistributionState.responses);
    },
    [
      isSpecificRouteSelection,
      orderedComparisonPeriodsKey,
      routeCarrierDistributionState.responses,
      routeCarrierDistributionState.status,
    ],
  );
  const hubComparative = useMemo(
    () => buildHubComparativeSeries(comparisonDatasets, routeFilters.carrier, hubOrigin),
    [comparisonDatasets, routeFilters.carrier, hubOrigin],
  );
  const hubFareStatsByPeriod = useMemo(
    () => {
      if (hubFareDistributionState.status !== "ready") {
        return [];
      }
      const periods = orderedComparisonPeriodsKey ? orderedComparisonPeriodsKey.split(",") : [];
      if (periods.length === 0) {
        return [];
      }
      return buildHubFareStatsByPeriod(periods, hubFareDistributionState.responses);
    },
    [hubFareDistributionState.responses, hubFareDistributionState.status, orderedComparisonPeriodsKey],
  );
  const hubTopCarrierShareDetails = useMemo(
    () => buildTopCarrierShareDetailsForHub(comparisonDatasets, routeFilters.carrier, hubOrigin),
    [comparisonDatasets, routeFilters.carrier, hubOrigin],
  );

  const routeSelectedPriceTrend = toTrendPoints(routeComparative, (point) => pickRoutePrice(point, routePriceMetric));
  const routeAvgFareTrend = toTrendPoints(routeComparative, (point) => point.avgFare);
  const routeMedianFareTrend = toTrendPoints(routeComparative, (point) => point.medianFare);
  const routeHhiTrend = toTrendPoints(routeComparative, (point) => point.avgHhi);
  const routePassengersTrend = toTrendPoints(routeComparative, (point) => point.totalPassengers);
  const routeShareTrend = toTrendPoints(routeComparative, (point) => point.marketSharePct);

  const hubAvgFareTrend = toTrendPoints(hubFareStatsByPeriod, (point) => point.avgFare);
  const hubMedianFareTrend = toTrendPoints(hubFareStatsByPeriod, (point) => point.medianFare);
  const hubIqrFareTrend = toTrendPoints(hubFareStatsByPeriod, (point) => point.iqrFare);
  const hubHhiTrend = toTrendPoints(hubComparative, (point) => point.avgHhi);
  const hubPassengersTrend = toTrendPoints(hubComparative, (point) => point.totalPassengers);
  const hubShareTrend = toTrendPoints(hubComparative, (point) => point.marketSharePct);
  const hubThroughputTrend = toTrendPoints(hubComparative, (point) => point.throughput);

  const routeSelectedPriceDelta = computeTrendDelta(routeSelectedPriceTrend, effectiveBasePeriod);
  const routeAvgFareDelta = computeTrendDelta(routeAvgFareTrend, effectiveBasePeriod);
  const routeMedianFareDelta = computeTrendDelta(routeMedianFareTrend, effectiveBasePeriod);
  const routeHhiDelta = computeTrendDelta(routeHhiTrend, effectiveBasePeriod);
  const routePassengerDelta = computeTrendDelta(routePassengersTrend, effectiveBasePeriod);
  const routeShareDelta = computeTrendDelta(routeShareTrend, effectiveBasePeriod);

  const hubAvgFareDelta = computeTrendDelta(hubAvgFareTrend, effectiveBasePeriod);
  const hubHhiDelta = computeTrendDelta(hubHhiTrend, effectiveBasePeriod);
  const hubPassengerDelta = computeTrendDelta(hubPassengersTrend, effectiveBasePeriod);
  const hubShareDelta = computeTrendDelta(hubShareTrend, effectiveBasePeriod);
  const hubThroughputDelta = computeTrendDelta(hubThroughputTrend, effectiveBasePeriod);
  const hubComparisonPeriodLabel = hubAvgFareDelta.endLabel
    ?? comparisonDatasets[comparisonDatasets.length - 1]?.period
    ?? "";
  const hubActivityDataset = useMemo(
    () => comparisonDatasets.find((dataset) => dataset.period === hubComparisonPeriodLabel)
      ?? comparisonDatasets[comparisonDatasets.length - 1]
      ?? null,
    [comparisonDatasets, hubComparisonPeriodLabel],
  );
  const hubCarrierActivityRows = useMemo<HubCarrierActivityRow[]>(() => {
    if (!hubActivityDataset || !isSpecificHubSelection) {
      return [];
    }
    const scopedRouteRows = (Array.isArray(hubActivityDataset.routeRows) ? hubActivityDataset.routeRows : [])
      .filter((row) => normalizeAirportCode(row.Origin) === selectedHubCode);
    if (scopedRouteRows.length === 0) {
      return [];
    }

    const byCarrier = new Map<string, {
      passengers: number;
      destinations: Set<string>;
      routes: Map<string, number>;
    }>();
    let totalPassengers = 0;

    scopedRouteRows.forEach((row) => {
      const carrierCode = normalizeCarrierCode(String(row.Carrier ?? ""));
      const destCode = normalizeAirportCode(row.Dest);
      const passengers = Number(row.total_passengers ?? 0);
      if (!carrierCode || !destCode || !Number.isFinite(passengers) || passengers <= 0) {
        return;
      }
      const routeKey = `${selectedHubCode} -> ${destCode}`;
      const current = byCarrier.get(carrierCode) ?? {
        passengers: 0,
        destinations: new Set<string>(),
        routes: new Map<string, number>(),
      };
      current.passengers += passengers;
      current.destinations.add(destCode);
      current.routes.set(routeKey, (current.routes.get(routeKey) ?? 0) + passengers);
      byCarrier.set(carrierCode, current);
      totalPassengers += passengers;
    });

    return Array.from(byCarrier.entries())
      .map(([carrierCode, values]) => {
        const topRouteEntry = Array.from(values.routes.entries())
          .sort((a, b) => b[1] - a[1])[0];
        return {
          carrierCode,
          carrierLabel: getCarrierDisplayName(carrierCode, carrierLookup),
          passengers: values.passengers,
          sharePct: totalPassengers > 0 ? (values.passengers / totalPassengers) * 100 : 0,
          destinationsServed: values.destinations.size,
          topRoute: topRouteEntry?.[0] ?? "-",
          topRoutePassengers: topRouteEntry?.[1] ?? 0,
        };
      })
      .sort((a, b) => b.passengers - a.passengers);
  }, [carrierLookup, hubActivityDataset, isSpecificHubSelection, selectedHubCode]);
  const hubTopAirlineBars = useMemo(
    () => hubCarrierActivityRows.slice(0, 10).map((row) => ({
      label: row.carrierLabel,
      value: row.passengers,
    })),
    [hubCarrierActivityRows],
  );
  const hubTopRouteBars = useMemo(() => {
    if (!hubActivityDataset || !isSpecificHubSelection) {
      return [];
    }
    const routePassengers = new Map<string, number>();
    const scopedRouteRows = (Array.isArray(hubActivityDataset.routeRows) ? hubActivityDataset.routeRows : [])
      .filter((row) => normalizeAirportCode(row.Origin) === selectedHubCode);
    scopedRouteRows.forEach((row) => {
      const destCode = normalizeAirportCode(row.Dest);
      const passengers = Number(row.total_passengers ?? 0);
      if (!destCode || !Number.isFinite(passengers) || passengers <= 0) {
        return;
      }
      const routeLabel = formatRouteDisplay(selectedHubCode, destCode);
      routePassengers.set(routeLabel, (routePassengers.get(routeLabel) ?? 0) + passengers);
    });
    return Array.from(routePassengers.entries())
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
  }, [hubActivityDataset, isSpecificHubSelection, selectedHubCode]);
  const hubActivitySummary = useMemo(() => {
    if (!hubActivityDataset || !isSpecificHubSelection) {
      return null;
    }
    const scopedRouteRows = (Array.isArray(hubActivityDataset.routeRows) ? hubActivityDataset.routeRows : [])
      .filter((row) => normalizeAirportCode(row.Origin) === selectedHubCode);
    if (scopedRouteRows.length === 0) {
      return null;
    }
    const carrierSet = new Set<string>();
    const destinationSet = new Set<string>();
    let totalPassengers = 0;
    scopedRouteRows.forEach((row) => {
      const carrierCode = normalizeCarrierCode(String(row.Carrier ?? ""));
      const destCode = normalizeAirportCode(row.Dest);
      const passengers = Number(row.total_passengers ?? 0);
      if (carrierCode) {
        carrierSet.add(carrierCode);
      }
      if (destCode) {
        destinationSet.add(destCode);
      }
      if (Number.isFinite(passengers) && passengers > 0) {
        totalPassengers += passengers;
      }
    });
    return {
      totalPassengers,
      carrierCount: carrierSet.size,
      destinationCount: destinationSet.size,
      topCarrierLabel: hubCarrierActivityRows[0]?.carrierLabel ?? "-",
      topCarrierSharePct: hubCarrierActivityRows[0]?.sharePct ?? 0,
      topRouteLabel: hubTopRouteBars[0]?.label ?? "-",
      topRoutePassengers: hubTopRouteBars[0]?.value ?? 0,
    };
  }, [hubActivityDataset, hubCarrierActivityRows, hubTopRouteBars, isSpecificHubSelection, selectedHubCode]);

  const routePriceByPeriodRows = buildPeriodValueChangeRows(routeSelectedPriceTrend, effectiveBasePeriod);
  const hubAvgFareByPeriodRows = buildPeriodValueChangeRows(hubAvgFareTrend, effectiveBasePeriod);
  const hubMedianFareByPeriodRows = buildPeriodValueChangeRows(hubMedianFareTrend, effectiveBasePeriod);
  const hubIqrFareByPeriodRows = buildPeriodValueChangeRows(hubIqrFareTrend, effectiveBasePeriod);
  const routeFareChangeSummary = `Largest route fare shifts from base to comparison period; minimum ${formatNumber(MIN_ROUTE_FARE_CHANGE_PASSENGERS)} passengers each period.`;
  const routeCarrierShareSummary = "Largest carrier share shifts from base to comparison period, measured in percentage points.";

  const routeFareChange = useMemo(
    () => buildRouteMarketFareChange(
      comparisonDatasets,
      routeFilters,
      8,
      effectiveBasePeriod,
      MIN_ROUTE_FARE_CHANGE_PASSENGERS,
    ),
    [comparisonDatasets, routeFilters, effectiveBasePeriod],
  );
  const routeShareShift = useMemo(
    () => buildRouteCarrierShareShift(comparisonDatasets, routeFilters, 8, effectiveBasePeriod),
    [comparisonDatasets, routeFilters, effectiveBasePeriod],
  );
  const hubPassengerChange = useMemo(
    () => buildHubPassengerChange(
      comparisonDatasets,
      routeFilters.carrier,
      hubOrigin,
      8,
      effectiveBasePeriod,
      MIN_HUB_PASSENGER_CHANGE_PASSENGERS,
    ),
    [comparisonDatasets, routeFilters.carrier, hubOrigin, effectiveBasePeriod],
  );
  const routeCarrierFrequencyBands: FareFrequencyBandSeries[] = useMemo(
    () => buildRouteFareFrequencyBands(comparisonDatasets, routeFilters),
    [comparisonDatasets, routeFilters],
  );
  const routeCarrierFrequencyShift = useMemo(
    () => buildFareFrequencyBandShift(routeCarrierFrequencyBands, 4, effectiveBasePeriod),
    [routeCarrierFrequencyBands, effectiveBasePeriod],
  );

  const latestRoute = routeComparative[routeComparative.length - 1];
  const latestHub = hubComparative[hubComparative.length - 1];
  const latestHubFareStats = hubFareStatsByPeriod[hubFareStatsByPeriod.length - 1];
  const latestHubTopContributors = formatCarrierContributors(latestHubFareStats?.topContributingCarriers ?? []);
  const hubOptions = useMemo(
    () => Array.from(new Set(
      allHubRows
        .map((row) => normalizeAirportCode(row.Origin))
        .filter((code) => code.length === 3),
    )).sort((a, b) => getAirportDisplayName(a).localeCompare(getAirportDisplayName(b))),
    [allHubRows],
  );

  function formatCarrierContributors(codes: string[]): string {
    if (!codes.length) {
      return "-";
    }
    const labels = codes.map((code) => getCarrierDisplayName(code, carrierLookup));
    if (labels.length <= 3) {
      return labels.join(", ");
    }
    return `${labels.slice(0, 3).join(", ")} +${labels.length - 3} more`;
  }

  function handleViewTabChange(key: string) {
    const next = key === "hubxairline" ? "hubxairline" : "routexairline";
    setActiveView(next);
    setRouteFilters(routeFilterDefaults);
    setHubOrigin("");
  }

  function toggleComparisonPeriod(period: string) {
    setComparisonPeriods((previous) => {
      const already = previous.includes(period);
      if (already) {
        if (previous.length <= MIN_COMPARE_PERIODS) {
          return previous;
        }
        return previous.filter((value) => value !== period);
      }
      return sortedPeriodKeys.filter((key) => key === period || previous.includes(key));
    });
  }

  function renderTopComparativeControls(): JSX.Element {
    return (
      <Card
        title="Compared Periods"
        subtitle={`Select ${MIN_COMPARE_PERIODS} or more periods and choose a base period for change calculations.`}
      >
        <div className="comparative-controls">
          <label>
            Base Period
            <select value={effectiveBasePeriod} onChange={(event) => setBaseComparisonPeriod(event.target.value)}>
              {comparisonPeriods.map((period) => (
                <option key={period} value={period}>
                  {period}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="compare-period-toggle-grid">
          {sortedPeriodKeys.map((period) => {
            const selected = comparisonPeriodSet.has(period);
            const lockRemoval = selected && comparisonPeriods.length <= MIN_COMPARE_PERIODS;
            return (
              <button
                key={period}
                type="button"
                className={`compare-period-toggle ${selected ? "compare-period-toggle--active" : ""}`}
                onClick={() => toggleComparisonPeriod(period)}
                disabled={lockRemoval}
                title={lockRemoval ? `Keep at least ${MIN_COMPARE_PERIODS} periods selected.` : ""}
              >
                {period}
              </button>
            );
          })}
        </div>
      </Card>
    );
  }

  function renderPriceByPeriodTable(
    rows: ReturnType<typeof buildPeriodValueChangeRows>,
    title: string,
    metricLabel: string,
    basePeriodLabel: string,
  ): JSX.Element {
    return (
      <Card title={title} subtitle={`${metricLabel} by period and percent change vs base period ${basePeriodLabel}`}>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Period</th>
                <th>{metricLabel}</th>
                <th>Change vs Base ($)</th>
                <th>Change vs Base (%)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.period}>
                  <td>{row.period}</td>
                  <td>{formatCurrency(row.value)}</td>
                  <td>{row.changeAbs === null ? "-" : formatCurrency(row.changeAbs)}</td>
                  <td>{row.changePct === null ? "-" : `${row.changePct.toFixed(1)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    );
  }

  function renderTopCarrierShareDetailInline(
    rows: TopCarrierShareDetail[],
    note: string,
  ): JSX.Element {
    return (
      <>
        <p className="top-share-inline-title">{note}</p>
        <div className="table-wrap">
          <table className="data-table top-share-inline-table">
            <thead>
              <tr>
                <th>Period</th>
                <th>Top Carrier Share</th>
                <th>Top Carrier(s) + Contribution</th>
                <th>Total Passengers</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`top-share-inline-${row.period}`}>
                  <td>{row.period}</td>
                  <td>{`${row.topSharePct.toFixed(1)}%`}</td>
                  <td>
                    {row.topCarriers.length === 0
                      ? "-"
                      : row.topCarriers.map((carrier) => (
                        <div key={`${row.period}-${carrier.carrier}`}>
                          {getCarrierDisplayName(carrier.carrier, carrierLookup)}: {carrier.sharePct.toFixed(1)}% ({formatNumber(carrier.passengers)} pax)
                        </div>
                      ))}
                  </td>
                  <td>{formatNumber(row.totalPassengers)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  }

  function buildContributionLines(
    delta: TrendDelta,
    byPeriod: Map<string, ContributionStats>,
    marketLabel: string,
  ): string[] {
    const lines: string[] = [];
    if (delta.startLabel) {
      const base = byPeriod.get(delta.startLabel);
      if (base) {
        lines.push(`Base contributing rows: ${formatNumber(base.rows)}`);
        lines.push(`Base ${marketLabel}: ${formatNumber(base.markets)}`);
        lines.push(`Base passengers: ${formatNumber(base.passengers)}`);
      }
    }
    if (delta.endLabel) {
      const compare = byPeriod.get(delta.endLabel);
      if (compare) {
        lines.push(`Comparison contributing rows: ${formatNumber(compare.rows)}`);
        lines.push(`Comparison ${marketLabel}: ${formatNumber(compare.markets)}`);
        lines.push(`Comparison passengers: ${formatNumber(compare.passengers)}`);
      }
    }
    return lines;
  }

  function describePriceMetric(metric: RoutePriceMetric): string {
    if (metric === "avg") {
      return "Passenger-weighted average fare = sum(fare x passengers) / sum(passengers).";
    }
    if (metric === "median") {
      return "Passenger-weighted median fare where cumulative passenger weight reaches 50%.";
    }
    if (metric === "min") {
      return "Minimum observed weighted fare across the current scope.";
    }
    return "Maximum observed weighted fare across the current scope.";
  }

  function renderRouteAllCarrierFareSection(): JSX.Element {
    if (!isSpecificRouteSelection) {
      return (
        <Card
          title="Route Carrier Fare Evolution Across Periods"
          subtitle="Select a specific Origin + Destination route to compare all carriers over time."
        >
          <p className="muted">This section activates when both origin and destination are specific airports.</p>
        </Card>
      );
    }

    if (routeCarrierDistributionState.status === "loading") {
      return (
        <Card
          title="Route Carrier Fare Evolution Across Periods"
          subtitle={selectedRouteLabel}
        >
          <p className="muted">Loading fare-distribution trends from DB1B source files...</p>
        </Card>
      );
    }

    if (routeCarrierDistributionState.status === "error") {
      return (
        <Card
          title="Route Carrier Fare Evolution Across Periods"
          subtitle={selectedRouteLabel}
        >
          <p className="muted">
            Could not load route fare-distribution trends.
            {" "}
            {routeCarrierDistributionState.error}
          </p>
        </Card>
      );
    }

    if (routeAllCarrierFareTrends.length === 0) {
      return (
        <Card
          title="Route Carrier Fare Evolution Across Periods"
          subtitle={selectedRouteLabel}
        >
          <p className="muted">No carrier fare history found for this route across selected periods.</p>
        </Card>
      );
    }

    const carrierSummaryRows = routeAllCarrierFareTrends.map((series) => {
      const avgDelta = computeTrendDelta(series.avgFareTrend, effectiveBasePeriod);
      const medianDelta = computeTrendDelta(series.medianFareTrend, effectiveBasePeriod);
      const iqrDelta = computeTrendDelta(series.iqrTrend, effectiveBasePeriod);
      const passengerDelta = computeTrendDelta(series.passengersTrend, effectiveBasePeriod);
      const shareDelta = computeTrendDelta(series.shareTrend, effectiveBasePeriod);
      return {
        carrier: series.carrier,
        carrierLabel: getCarrierDisplayName(series.carrier, carrierLookup, series.carrierName),
        avgNow: avgDelta.end,
        avgDelta,
        medianNow: medianDelta.end,
        medianDelta,
        iqrNow: iqrDelta.end,
        iqrDelta,
        passengersNow: passengerDelta.end,
        passengerDelta,
        shareNow: shareDelta.end,
        shareDelta,
        periodsWithData: series.periodsWithData,
        avgFareTrend: series.avgFareTrend,
        medianFareTrend: series.medianFareTrend,
        iqrTrend: series.iqrTrend,
        passengersTrend: series.passengersTrend,
        shareTrend: series.shareTrend,
      };
    });

    const selectedCarrierCode = normalizeCarrierCode(routeFilters.carrier);
    const topThree = carrierSummaryRows.slice(0, 3);
    const selectedCarrierRow = carrierSummaryRows.find((row) => row.carrier === selectedCarrierCode) ?? null;
    const collapsedRows = selectedCarrierRow && !topThree.some((row) => row.carrier === selectedCarrierRow.carrier)
      ? [...topThree, selectedCarrierRow]
      : topThree;
    const visibleRows = showAllRouteCarriers ? carrierSummaryRows : collapsedRows;

    return (
      <section className="panel-grid">
        <Card
          title="Route Carrier Fare Evolution Across Periods"
          subtitle={`${selectedRouteLabel}. Click an airline row to expand Avg, Median, IQR, Passenger, and Carrier Share trends.`}
        >
          <div className="carrier-accordion-toolbar">
            <p className="muted">
              {showAllRouteCarriers
                ? "Showing all carriers on this route."
                : "Showing top 3 carriers by passenger volume (plus selected carrier if different)."}
            </p>
            {routeCarrierDistributionState.missingPeriods.length > 0 ? (
              <p className="muted">
                Missing DB1B fare-distribution data for:
                {" "}
                {routeCarrierDistributionState.missingPeriods.join(", ")}
              </p>
            ) : null}
            {carrierSummaryRows.length > 3 ? (
              <AppButton
                variant="ghost"
                type="button"
                onClick={() => setShowAllRouteCarriers((previous) => !previous)}
              >
                {showAllRouteCarriers ? "Show Top 3" : "Show All Carriers"}
              </AppButton>
            ) : null}
          </div>
        </Card>

        <div className="carrier-accordion-list">
          {visibleRows.map((row, index) => {
            const color = trendPalette[index % trendPalette.length];
            const openBySelection = Boolean(selectedCarrierCode) && row.carrier === selectedCarrierCode;
            return (
              <details
                key={`carrier-trend-${row.carrier}`}
                className="carrier-accordion"
                {...(openBySelection ? { open: true } : {})}
              >
                <summary className="carrier-accordion__summary">
                  <span className="carrier-accordion__title">{row.carrierLabel}</span>
                  <span className="carrier-accordion__meta">
                    Avg {formatDeltaCell(row.avgDelta, formatCurrency)} | Median {formatDeltaCell(row.medianDelta, formatCurrency)} | IQR {formatDeltaCell(row.iqrDelta, formatCurrency)}
                  </span>
                </summary>
                <div className="carrier-accordion__content">
                  <div className="metrics-grid">
                    <MetricCard
                      label="Avg Fare"
                      value={formatCurrency(row.avgNow)}
                      hint={formatDeltaHint(row.avgDelta, formatCurrency)}
                    />
                    <MetricCard
                      label="Median Fare"
                      value={formatCurrency(row.medianNow)}
                      hint={formatDeltaHint(row.medianDelta, formatCurrency)}
                    />
                    <MetricCard
                      label="IQR"
                      value={formatCurrency(row.iqrNow)}
                      hint={formatDeltaHint(row.iqrDelta, formatCurrency)}
                    />
                    <MetricCard
                      label="Passengers"
                      value={formatNumber(row.passengersNow)}
                      hint={formatDeltaHint(row.passengerDelta, formatNumber)}
                    />
                    <MetricCard
                      label="Carrier Share"
                      value={`${row.shareNow.toFixed(1)}%`}
                      hint={formatDeltaHint(row.shareDelta, (value) => `${value.toFixed(1)}%`)}
                    />
                    <MetricCard
                      label="Periods With Data"
                      value={formatNumber(row.periodsWithData)}
                    />
                  </div>

                  <div className="two-col">
                    <LineTrendChart
                      title={`${row.carrierLabel} Avg Fare Trend`}
                      points={row.avgFareTrend}
                      color={color}
                      valueFormatter={formatCurrency}
                    />
                    <LineTrendChart
                      title={`${row.carrierLabel} Median Fare Trend`}
                      points={row.medianFareTrend}
                      color="#1f4e79"
                      valueFormatter={formatCurrency}
                    />
                    <LineTrendChart
                      title={`${row.carrierLabel} IQR Trend`}
                      subtitle="Interquartile range (Q3 - Q1)"
                      points={row.iqrTrend}
                      color="#3f7f63"
                      valueFormatter={formatCurrency}
                    />
                    <LineTrendChart
                      title={`${row.carrierLabel} Passenger Trend`}
                      points={row.passengersTrend}
                      color="#805ad5"
                      valueFormatter={formatNumber}
                    />
                    <LineTrendChart
                      title={`${row.carrierLabel} Carrier Share Trend`}
                      subtitle="Share of route passengers for this airline."
                      points={row.shareTrend}
                      color="#8b3d3d"
                      valueFormatter={(value) => `${value.toFixed(1)}%`}
                    />
                  </div>
                </div>
              </details>
            );
          })}
        </div>
      </section>
    );
  }

  function renderCarrierFrequencySection(
    title: string,
    subtitle: string,
    frequencyBands: FareFrequencyBandSeries[],
    frequencyShift: { increases: ComparativeChangeRow[]; decreases: ComparativeChangeRow[] },
  ): JSX.Element {
    if (!routeFilters.carrier) {
      return (
        <Card title={title} subtitle={subtitle}>
          <p className="muted">Select a carrier in filters to unlock fare-frequency comparison by price band.</p>
        </Card>
      );
    }

    if (frequencyBands.length === 0) {
      return (
        <Card title={title} subtitle={subtitle}>
          <p className="muted">No fare-frequency data available for the current selection.</p>
        </Card>
      );
    }

    const frequencyIncreaseRows = toBarRows(frequencyShift.increases);
    const frequencyDecreaseRows = toBarRows(frequencyShift.decreases);
    const frequencyShiftMax = getSharedBarMax([frequencyIncreaseRows, frequencyDecreaseRows]);

    return (
      <section className="panel-grid">
        <Card title={title} subtitle={`${subtitle} (${routeFilters.carrier.toUpperCase()})`}>
          <p className="muted">Each line shows passenger share (%) in a fare band across selected periods.</p>
        </Card>

        <div className="two-col">
          {frequencyBands.map((band, index) => (
            <LineTrendChart
              key={band.band}
              title={`${band.band} Fare Share Trend`}
              points={band.points}
              color={trendPalette[index % trendPalette.length]}
              valueFormatter={(value) => `${value.toFixed(1)}%`}
            />
          ))}
        </div>

        <div className="two-col">
          <SimpleBarChart
            title="Fare Band Frequency Gains vs Base (percentage points)"
            rows={frequencyIncreaseRows}
            maxValue={frequencyShiftMax}
            valueFormatter={(value) => `${value.toFixed(1)} percentage points`}
            color="#1f4e79"
          />
          <SimpleBarChart
            title="Fare Band Frequency Losses vs Base (percentage points)"
            rows={frequencyDecreaseRows}
            maxValue={frequencyShiftMax}
            valueFormatter={(value) => `${value.toFixed(1)} percentage points`}
            color="#8b3d3d"
          />
        </div>
      </section>
    );
  }

  function renderHubCarrierActivitySection(): JSX.Element | null {
    if (!isSpecificHubSelection) {
      return null;
    }

    const periodLabel = hubActivityDataset?.period ?? hubComparisonPeriodLabel ?? "-";
    if (hubCarrierActivityRows.length === 0) {
      return (
        <Card
          title="Hub Carrier & Route Activity"
          subtitle={`${selectedHubLabel} | ${periodLabel}`}
        >
          <p className="muted">No route-level activity found for this hub in the comparison period.</p>
        </Card>
      );
    }

    const hubActivityBarsMax = getSharedBarMax([hubTopAirlineBars, hubTopRouteBars]);

    return (
      <section className="panel-grid">
        <Card
          title="Hub Carrier & Route Activity"
          subtitle={`${selectedHubLabel} | ${periodLabel}`}
        >
          <p className="muted">
            Carrier mix plus route concentration for the selected hub in this comparison period.
          </p>
          {hubActivitySummary ? (
            <div className="hub-activity-summary-grid">
              <div className="hub-activity-pill">
                <span className="hub-activity-pill__label">Total Passengers</span>
                <span className="hub-activity-pill__value">{formatNumber(hubActivitySummary.totalPassengers)}</span>
              </div>
              <div className="hub-activity-pill">
                <span className="hub-activity-pill__label">Active Carriers</span>
                <span className="hub-activity-pill__value">{formatNumber(hubActivitySummary.carrierCount)}</span>
              </div>
              <div className="hub-activity-pill">
                <span className="hub-activity-pill__label">Destinations Served</span>
                <span className="hub-activity-pill__value">{formatNumber(hubActivitySummary.destinationCount)}</span>
              </div>
              <div className="hub-activity-pill">
                <span className="hub-activity-pill__label">Largest Carrier</span>
                <span className="hub-activity-pill__value">
                  {hubActivitySummary.topCarrierLabel} ({hubActivitySummary.topCarrierSharePct.toFixed(1)}%)
                </span>
              </div>
              <div className="hub-activity-pill">
                <span className="hub-activity-pill__label">Top Route</span>
                <span className="hub-activity-pill__value">{hubActivitySummary.topRouteLabel}</span>
                <span className="hub-activity-pill__meta">{formatNumber(hubActivitySummary.topRoutePassengers)} pax</span>
              </div>
            </div>
          ) : null}
        </Card>

        <div className="two-col">
          <SimpleBarChart
            title="Top Airlines at Selected Hub (Passengers)"
            rows={hubTopAirlineBars}
            maxValue={hubActivityBarsMax}
            valueFormatter={formatNumber}
            color="#1f4e79"
          />
          <SimpleBarChart
            title="Top Overall Routes from Selected Hub (Passengers)"
            rows={hubTopRouteBars}
            maxValue={hubActivityBarsMax}
            valueFormatter={formatNumber}
            color="#2f855a"
          />
        </div>

        <Card
          title="Carrier Mix at Selected Hub"
          subtitle="Each carrier's share and leading route. Top-routes chart above shows overall hub routes."
        >
          <DataTable
            rows={hubCarrierActivityRows}
            columns={hubCarrierActivityColumns}
            rowKey={(row) => `hub-activity-${periodLabel}-${row.carrierCode}`}
            className="hub-activity-table-wrap"
            tableClassName="hub-activity-table"
          />
        </Card>
      </section>
    );
  }

  function renderRouteComparative(): JSX.Element {
    if (!hasEnoughComparisonPeriods) {
      return (
        <section className="panel-grid">
          <EmptyState
            title="Need more periods selected"
            description={`Select at least ${MIN_COMPARE_PERIODS} periods to compute comparative metrics.`}
          />
        </section>
      );
    }

    const routeFareIncreaseRows = toBarRows(routeFareChange.increases);
    const routeFareDecreaseRows = toBarRows(routeFareChange.decreases);
    const routeFareChangeMax = getSharedBarMax([routeFareIncreaseRows, routeFareDecreaseRows]);
    const routeShareGainRows = toBarRows(routeShareShift.increases).map((row) => ({
      ...row,
      label: getCarrierDisplayName(row.label, carrierLookup),
    }));
    const routeShareLossRows = toBarRows(routeShareShift.decreases).map((row) => ({
      ...row,
      label: getCarrierDisplayName(row.label, carrierLookup),
    }));
    const routeShareShiftMax = getSharedBarMax([routeShareGainRows, routeShareLossRows]);

    return (
      <section className="panel-grid">
        {renderPriceByPeriodTable(
          routePriceByPeriodRows,
          "Route Price by Period",
          routePriceMetricLabels[routePriceMetric],
          effectiveBasePeriod,
        )}

        <div className="metrics-grid">
          <MetricCard
            label={routePriceMetricLabels[routePriceMetric]}
            value={formatCurrency(latestRoute ? pickRoutePrice(latestRoute, routePriceMetric) : 0)}
            hint={formatDeltaHint(routeSelectedPriceDelta, formatCurrency)}
            tooltip={buildMetricTooltip(
              routePriceMetricLabels[routePriceMetric],
              describePriceMetric(routePriceMetric),
              routeSelectedPriceDelta,
              formatCurrency,
              buildContributionLines(routeSelectedPriceDelta, routeContributionByPeriod, "route markets"),
            )}
          />
          <MetricCard
            label="Avg Fare"
            value={formatCurrency(latestRoute?.avgFare ?? 0)}
            hint={formatDeltaHint(routeAvgFareDelta, formatCurrency)}
            tooltip={buildMetricTooltip(
              "Avg Fare",
              describePriceMetric("avg"),
              routeAvgFareDelta,
              formatCurrency,
              buildContributionLines(routeAvgFareDelta, routeContributionByPeriod, "route markets"),
            )}
          />
          <MetricCard
            label="Median Fare"
            value={formatCurrency(latestRoute?.medianFare ?? 0)}
            hint={formatDeltaHint(routeMedianFareDelta, formatCurrency)}
            tooltip={buildMetricTooltip(
              "Median Fare",
              describePriceMetric("median"),
              routeMedianFareDelta,
              formatCurrency,
              buildContributionLines(routeMedianFareDelta, routeContributionByPeriod, "route markets"),
            )}
          />
          <MetricCard
            label="Passengers"
            value={formatNumber(latestRoute?.totalPassengers ?? 0)}
            hint={formatDeltaHint(routePassengerDelta, formatNumber)}
            tooltip={buildMetricTooltip(
              "Passengers",
              "Total passengers summed across all filtered route rows in each selected period.",
              routePassengerDelta,
              formatNumber,
              buildContributionLines(routePassengerDelta, routeContributionByPeriod, "route markets"),
            )}
          />
          <MetricCard
            label="Avg Route HHI"
            value={formatNumber(latestRoute?.avgHhi ?? 0)}
            hint={formatDeltaHint(routeHhiDelta, formatNumber)}
            tooltip={buildMetricTooltip(
              "Avg Route HHI",
              "Average HHI across unique filtered route markets in each period.",
              routeHhiDelta,
              formatNumber,
              buildContributionLines(routeHhiDelta, routeContributionByPeriod, "route markets"),
            )}
          />
          <MetricCard
            label="Top Carrier Share"
            value={`${(latestRoute?.marketSharePct ?? 0).toFixed(1)}%`}
            hint={formatDeltaHint(routeShareDelta, (value) => `${value.toFixed(1)}%`)}
            tooltip={buildMetricTooltip(
              "Top Carrier Share",
              "Passenger share of the largest carrier within the filtered route scope.",
              routeShareDelta,
              (value) => `${value.toFixed(1)}%`,
              buildContributionLines(routeShareDelta, routeContributionByPeriod, "route markets"),
            )}
          />
        </div>

        <div className="two-col">
          <LineTrendChart title={`${routePriceMetricLabels[routePriceMetric]} Trend`} points={routeSelectedPriceTrend} valueFormatter={formatCurrency} />
          <LineTrendChart title="Passenger Trend" points={routePassengersTrend} color="#1f4e79" valueFormatter={formatNumber} />
          <LineTrendChart title="HHI Trend" points={routeHhiTrend} color="#8b3d3d" valueFormatter={formatNumber} />
          <LineTrendChart
            title="Top Carrier Share Trend"
            subtitle={isSpecificRouteSelection
              ? "Percent of passengers on the selected route flown by the single largest carrier each period."
              : "Percent of passengers in the current route filter scope flown by the single largest carrier each period."}
            points={routeShareTrend}
            color="#3f7f63"
            valueFormatter={(value) => `${value.toFixed(1)}%`}
            footer={renderTopCarrierShareDetailInline(
              routeTopCarrierShareDetails,
              "Carrier(s) contributing to each period's top-share point.",
            )}
          />
        </div>

        <div className="two-col">
          <SimpleBarChart
            title="Routes With Largest Fare Increases vs Base"
            subtitle={routeFareChangeSummary}
            rows={routeFareIncreaseRows}
            maxValue={routeFareChangeMax}
            valueFormatter={formatCurrency}
            color="#b7791f"
          />
          <SimpleBarChart
            title="Routes With Largest Fare Decreases vs Base"
            subtitle={routeFareChangeSummary}
            rows={routeFareDecreaseRows}
            maxValue={routeFareChangeMax}
            valueFormatter={formatCurrency}
            color="#2f855a"
          />
          <SimpleBarChart
            title="Carrier Share Gains vs Base (percentage points)"
            subtitle={routeCarrierShareSummary}
            rows={routeShareGainRows}
            maxValue={routeShareShiftMax}
            valueFormatter={(value) => `${value.toFixed(1)} percentage points`}
            color="#1f4e79"
          />
          <SimpleBarChart
            title="Carrier Share Losses vs Base (percentage points)"
            subtitle={routeCarrierShareSummary}
            rows={routeShareLossRows}
            maxValue={routeShareShiftMax}
            valueFormatter={(value) => `${value.toFixed(1)} percentage points`}
            color="#8b3d3d"
          />
        </div>

        {isSpecificRouteSelection
          ? renderRouteAllCarrierFareSection()
          : renderCarrierFrequencySection(
            "Carrier Fare Frequency Across Periods",
            "Route carrier selection",
            routeCarrierFrequencyBands,
            routeCarrierFrequencyShift,
          )}
      </section>
    );
  }

  function renderHubComparative(): JSX.Element {
    if (!hasEnoughComparisonPeriods) {
      return (
        <section className="panel-grid">
          <EmptyState
            title="Need more periods selected"
            description={`Select at least ${MIN_COMPARE_PERIODS} periods to compute comparative metrics.`}
          />
        </section>
      );
    }

    const hasHubFareStats = hubFareStatsByPeriod.length > 0;
    const hubPassengerIncreaseRows = toBarRows(hubPassengerChange.increases);
    const hubPassengerDecreaseRows = toBarRows(hubPassengerChange.decreases);
    const hubPassengerChangeMax = getSharedBarMax([hubPassengerIncreaseRows, hubPassengerDecreaseRows]);

    return (
      <section className="panel-grid">
        <div className="metrics-grid">
          <MetricCard
            label="Passengers"
            value={formatNumber(latestHub?.totalPassengers ?? 0)}
            hint={formatDeltaHint(hubPassengerDelta, formatNumber)}
            tooltip={buildMetricTooltip(
              "Passengers",
              "Total passengers summed across all filtered hub rows in each selected period.",
              hubPassengerDelta,
              formatNumber,
              buildContributionLines(hubPassengerDelta, hubContributionByPeriod, "hubs"),
            )}
          />
          <MetricCard
            label="Avg Hub HHI"
            value={formatNumber(latestHub?.avgHhi ?? 0)}
            hint={formatDeltaHint(hubHhiDelta, formatNumber)}
            tooltip={buildMetricTooltip(
              "Avg Hub HHI",
              "Average HHI across unique filtered hubs in each period.",
              hubHhiDelta,
              formatNumber,
              buildContributionLines(hubHhiDelta, hubContributionByPeriod, "hubs"),
            )}
          />
          <MetricCard
            label="Top Carrier Share"
            value={`${(latestHub?.marketSharePct ?? 0).toFixed(1)}%`}
            hint={formatDeltaHint(hubShareDelta, (value) => `${value.toFixed(1)}%`)}
            tooltip={buildMetricTooltip(
              "Top Carrier Share",
              "Passenger share of the largest carrier within the filtered hub scope.",
              hubShareDelta,
              (value) => `${value.toFixed(1)}%`,
              buildContributionLines(hubShareDelta, hubContributionByPeriod, "hubs"),
            )}
          />
          <MetricCard
            label="Throughput"
            value={formatNumber(latestHub?.throughput ?? 0)}
            hint={formatDeltaHint(hubThroughputDelta, formatNumber)}
            tooltip={buildMetricTooltip(
              "Throughput",
              "Sum of passenger-distance across filtered hub rows (passengers x distance).",
              hubThroughputDelta,
              formatNumber,
              buildContributionLines(hubThroughputDelta, hubContributionByPeriod, "hubs"),
            )}
          />
        </div>

        <div className="two-col">
          <LineTrendChart title="Passenger Trend" points={hubPassengersTrend} color="#1f4e79" valueFormatter={formatNumber} />
          <LineTrendChart title="HHI Trend" points={hubHhiTrend} color="#8b3d3d" valueFormatter={formatNumber} />
          <LineTrendChart
            title="Top Carrier Share Trend"
            subtitle={hubOrigin
              ? `Percent of passengers at ${getAirportDisplayName(hubOrigin)} flown by the single largest carrier each period.`
              : "Percent of passengers in the current hub filter scope flown by the single largest carrier each period."}
            points={hubShareTrend}
            color="#3f7f63"
            valueFormatter={(value) => `${value.toFixed(1)}%`}
            footer={renderTopCarrierShareDetailInline(
              hubTopCarrierShareDetails,
              "Carrier(s) contributing to each period's top-share point.",
            )}
          />
          <LineTrendChart title="Throughput Trend" points={hubThroughputTrend} color="#5f3dc4" valueFormatter={formatNumber} />
        </div>

        {!isSpecificHubSelection ? (
          <div className="two-col">
            <SimpleBarChart
              title="Hubs With Largest Passenger Increases vs Base"
              subtitle={`Includes only hubs with >= ${formatNumber(MIN_HUB_PASSENGER_CHANGE_PASSENGERS)} passengers in both base and comparison periods.`}
              rows={hubPassengerIncreaseRows}
              maxValue={hubPassengerChangeMax}
              valueFormatter={formatNumber}
              color="#1f4e79"
            />
            <SimpleBarChart
              title="Hubs With Largest Passenger Decreases vs Base"
              subtitle={`Includes only hubs with >= ${formatNumber(MIN_HUB_PASSENGER_CHANGE_PASSENGERS)} passengers in both base and comparison periods.`}
              rows={hubPassengerDecreaseRows}
              maxValue={hubPassengerChangeMax}
              valueFormatter={formatNumber}
              color="#8b3d3d"
            />
          </div>
        ) : null}

        {renderHubCarrierActivitySection()}

        {hubFareDistributionState.status === "loading" ? (
          <Card title="Hub Fare Metrics (DB1B)" subtitle={selectedHubLabel}>
            <p className="muted">Loading hub fare distributions directly from DB1B-derived fare bins...</p>
          </Card>
        ) : null}

        {hubFareDistributionState.status === "error" ? (
          <Card title="Hub Fare Metrics (DB1B)" subtitle={selectedHubLabel}>
            <p className="muted">
              Could not load hub fare distributions.
              {" "}
              {hubFareDistributionState.error}
            </p>
          </Card>
        ) : null}

        {hubFareDistributionState.missingPeriods.length > 0 ? (
          <Card title="Hub Fare Metrics (DB1B)" subtitle={selectedHubLabel}>
            <p className="muted">
              Missing DB1B fare-distribution data for:
              {" "}
              {hubFareDistributionState.missingPeriods.join(", ")}
            </p>
          </Card>
        ) : null}

        {hasHubFareStats ? (
          <div className="three-col">
            {renderPriceByPeriodTable(
              hubAvgFareByPeriodRows,
              "Hub Average Fare by Period",
              "Avg Fare",
              effectiveBasePeriod,
            )}
            {renderPriceByPeriodTable(
              hubMedianFareByPeriodRows,
              "Hub Median Fare by Period",
              "Median Fare",
              effectiveBasePeriod,
            )}
            {renderPriceByPeriodTable(
              hubIqrFareByPeriodRows,
              "Hub IQR Fare by Period",
              "IQR Fare",
              effectiveBasePeriod,
            )}
          </div>
        ) : null}

        {hasHubFareStats ? (
          <div className="two-col">
            <LineTrendChart
              title="Avg Fare Trend"
              subtitle={`Passenger-weighted from DB1B fare bins. Latest largest contributors: ${latestHubTopContributors}.`}
              points={hubAvgFareTrend}
              valueFormatter={formatCurrency}
            />
            <LineTrendChart
              title="Median Fare Trend"
              subtitle={`50th percentile from DB1B fare bins. Latest largest contributors: ${latestHubTopContributors}.`}
              points={hubMedianFareTrend}
              color="#1f4e79"
              valueFormatter={formatCurrency}
            />
            <LineTrendChart
              title="IQR Fare Trend"
              subtitle={`Interquartile range (Q3 - Q1) from DB1B fare bins. Latest largest contributors: ${latestHubTopContributors}.`}
              points={hubIqrFareTrend}
              color="#2f855a"
              valueFormatter={formatCurrency}
            />
          </div>
        ) : null}

      </section>
    );
  }

  return (
    <PageShell
      title="Multi-Period Analytics"
      subtitle={`Periods: ${sortedDatasets.map((dataset) => dataset.period).join(", ")}`}
    >
      <RouteFilterBar
        filters={routeFilters}
        onChange={setRouteFilters}
        period={filterPeriodLabel}
        rows={allRouteRows}
        hubOptions={activeView === "hubxairline" ? hubOptions : undefined}
        hubValue={activeView === "hubxairline" ? hubOrigin : ""}
        onHubChange={activeView === "hubxairline" ? setHubOrigin : undefined}
        priceMetricValue={activePeriodTab === COMPARATIVE_TAB_KEY && activeView === "routexairline"
          ? routePriceMetric
          : undefined}
        priceMetricOptions={activePeriodTab === COMPARATIVE_TAB_KEY && activeView === "routexairline"
          ? priceMetricOptions
          : undefined}
        onPriceMetricChange={activePeriodTab === COMPARATIVE_TAB_KEY && activeView === "routexairline"
          ? (nextPriceMetric) => {
            const typedMetric = nextPriceMetric as RoutePriceMetric;
            setRoutePriceMetric(typedMetric);
          }
          : undefined}
        showPeriod={activePeriodTab !== COMPARATIVE_TAB_KEY}
        showOrigin={activeView === "routexairline"}
        showDestination={activeView === "routexairline"}
      />

      {activePeriodTab === COMPARATIVE_TAB_KEY ? renderTopComparativeControls() : null}

      <Tabs
        className="tabs--mode"
        options={[
          { key: "routexairline", label: "RouteXAirline" },
          { key: "hubxairline", label: "HubXAirline" },
        ]}
        activeKey={activeView}
        onChange={handleViewTabChange}
      />

      <Tabs
        className="tabs--period"
        options={periodTabs}
        activeKey={activePeriodTab}
        onChange={setActivePeriodTab}
      />

      {activePeriodTab !== COMPARATIVE_TAB_KEY ? (
        selectedDataset ? (
          activeView === "routexairline" ? (
            <>
              <MarketOverviewPanel rows={selectedRouteRows} filters={routeFilters} />
              <RouteHubInsightsPanel
                period={selectedDataset.period}
                routeRows={selectedRouteRows}
                hubRows={selectedHubRows}
                routeFilters={routeFilters}
                view="routexairline"
              />
            </>
          ) : (
            <RouteHubInsightsPanel
              period={selectedDataset.period}
              routeRows={selectedRouteRowsForHub}
              hubRows={selectedHubRows}
              routeFilters={{ origin: hubOrigin, dest: "", carrier: routeFilters.carrier }}
              view="hubxairline"
            />
          )
        ) : (
          <EmptyState title="Period not found" description="Choose another period tab or reload selected periods." />
        )
      ) : (
        activeView === "routexairline" ? renderRouteComparative() : renderHubComparative()
      )}
    </PageShell>
  );
}

/**
 * @file src/features/results/comparative.ts
 * @description Comparative period-to-period metric helpers for multi-period analytics views.
 */

import type { DatasetRecord, HubMarketPowerRow, RouteFilters, RouteMarketPowerRow } from "../../types/data";
import { getAirportDisplayName, normalizeAirportCode } from "../../utils/airports";
import { applyHubFilters, applyRouteFilters } from "./analytics";

export type RoutePriceMetric = "avg" | "max" | "min" | "median";

export interface RouteComparativePoint {
  period: string;
  totalPassengers: number;
  avgFare: number;
  minFare: number;
  maxFare: number;
  medianFare: number;
  avgHhi: number;
  marketSharePct: number;
}

export interface HubComparativePoint {
  period: string;
  totalPassengers: number;
  avgFare: number;
  minFare: number;
  maxFare: number;
  medianFare: number;
  avgHhi: number;
  marketSharePct: number;
  throughput: number;
}

export interface TrendPoint {
  label: string;
  value: number;
}

interface WeightedValue {
  value: number;
  weight: number;
}

function isFinitePositive(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function computeWeightedMedian(items: WeightedValue[]): number {
  if (items.length === 0) {
    return 0;
  }
  const sorted = [...items].sort((a, b) => a.value - b.value);
  const totalWeight = sorted.reduce((sum, item) => sum + item.weight, 0);
  if (!isFinitePositive(totalWeight)) {
    const middle = Math.floor(sorted.length / 2);
    return sorted[middle]?.value ?? 0;
  }
  const target = totalWeight / 2;
  let running = 0;
  for (const item of sorted) {
    running += item.weight;
    if (running >= target) {
      return item.value;
    }
  }
  return sorted[sorted.length - 1]?.value ?? 0;
}

function periodOrderValue(period: string): number {
  const match = /^(\d{4})_Q([1-4])$/.exec(period);
  if (!match) {
    return Number.MAX_SAFE_INTEGER;
  }
  return Number(match[1]) * 10 + Number(match[2]);
}

export function sortDatasetsByPeriod(datasets: DatasetRecord[]): DatasetRecord[] {
  return [...datasets].sort((a, b) => periodOrderValue(a.period) - periodOrderValue(b.period));
}

function computeAverageHhi(rows: RouteMarketPowerRow[]): number {
  const routeHhi = new Map<string, number>();
  rows.forEach((row) => {
    const key = `${row.Origin}_${row.Dest}`;
    if (!routeHhi.has(key) && Number.isFinite(row.route_HHI)) {
      routeHhi.set(key, Number(row.route_HHI));
    }
  });
  const values = Array.from(routeHhi.values());
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeTopCarrierSharePct(rows: RouteMarketPowerRow[]): number {
  if (rows.length === 0) {
    return 0;
  }
  const byCarrier = new Map<string, number>();
  let totalPassengers = 0;
  rows.forEach((row) => {
    const carrier = String(row.Carrier ?? "").trim().toUpperCase();
    const passengers = Number(row.total_passengers ?? 0);
    if (!carrier || !Number.isFinite(passengers) || passengers <= 0) {
      return;
    }
    byCarrier.set(carrier, (byCarrier.get(carrier) ?? 0) + passengers);
    totalPassengers += passengers;
  });
  if (!isFinitePositive(totalPassengers) || byCarrier.size === 0) {
    return 0;
  }
  const largestCarrierPassengers = Math.max(...Array.from(byCarrier.values()));
  return (largestCarrierPassengers / totalPassengers) * 100;
}

function computeTopCarrierSharePctForHub(rows: HubMarketPowerRow[]): number {
  if (rows.length === 0) {
    return 0;
  }
  const byCarrier = new Map<string, number>();
  let totalPassengers = 0;
  rows.forEach((row) => {
    const carrier = String(row.Carrier ?? "").trim().toUpperCase();
    const passengers = Number(row.total_passengers ?? 0);
    if (!carrier || !Number.isFinite(passengers) || passengers <= 0) {
      return;
    }
    byCarrier.set(carrier, (byCarrier.get(carrier) ?? 0) + passengers);
    totalPassengers += passengers;
  });
  if (!isFinitePositive(totalPassengers) || byCarrier.size === 0) {
    return 0;
  }
  const largestCarrierPassengers = Math.max(...Array.from(byCarrier.values()));
  return (largestCarrierPassengers / totalPassengers) * 100;
}

function computeAverageHubHhi(rows: HubMarketPowerRow[]): number {
  const byHub = new Map<string, number>();
  rows.forEach((row) => {
    const key = normalizeAirportCode(row.Origin);
    if (!key) {
      return;
    }
    if (!byHub.has(key) && Number.isFinite(row.hub_HHI)) {
      byHub.set(key, Number(row.hub_HHI));
    }
  });
  const values = Array.from(byHub.values());
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeRouteComparativePoint(period: string, rows: RouteMarketPowerRow[]): RouteComparativePoint {
  if (rows.length === 0) {
    return {
      period,
      totalPassengers: 0,
      avgFare: 0,
      minFare: 0,
      maxFare: 0,
      medianFare: 0,
      avgHhi: 0,
      marketSharePct: 0,
    };
  }

  let totalPassengers = 0;
  let fareXPassengers = 0;
  let minFare = Number.POSITIVE_INFINITY;
  let maxFare = Number.NEGATIVE_INFINITY;
  const weightedFares: WeightedValue[] = [];

  rows.forEach((row) => {
    const passengers = Number(row.total_passengers ?? 0);
    const fare = Number(row.avg_fare_weighted ?? 0);
    if (!isFinitePositive(passengers) || !isFinitePositive(fare)) {
      return;
    }
    totalPassengers += passengers;
    fareXPassengers += fare * passengers;
    minFare = Math.min(minFare, fare);
    maxFare = Math.max(maxFare, fare);
    weightedFares.push({ value: fare, weight: passengers });
  });

  const avgFare = totalPassengers > 0 ? fareXPassengers / totalPassengers : 0;
  const medianFare = computeWeightedMedian(weightedFares);
  const avgHhi = computeAverageHhi(rows);
  const marketSharePct = computeTopCarrierSharePct(rows);

  return {
    period,
    totalPassengers,
    avgFare,
    minFare: Number.isFinite(minFare) ? minFare : 0,
    maxFare: Number.isFinite(maxFare) ? maxFare : 0,
    medianFare,
    avgHhi,
    marketSharePct,
  };
}

export function buildRouteComparativeSeries(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
): RouteComparativePoint[] {
  return sortDatasetsByPeriod(datasets).map((dataset) => {
    const rows = applyRouteFilters(Array.isArray(dataset.routeRows) ? dataset.routeRows : [], routeFilters);
    return computeRouteComparativePoint(dataset.period, rows);
  });
}

export function buildHubComparativeSeries(
  datasets: DatasetRecord[],
  carrierFilter: string,
  hubOrigin = "",
): HubComparativePoint[] {
  return sortDatasetsByPeriod(datasets).map((dataset) => {
    const scopedRows = applyHubFilters(Array.isArray(dataset.hubRows) ? dataset.hubRows : [], { carrier: carrierFilter });
    const origin = hubOrigin.trim().toUpperCase();
    const rows = origin
      ? scopedRows.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === origin)
      : scopedRows;
    let totalPassengers = 0;
    let fareXPassengers = 0;
    let minFare = Number.POSITIVE_INFINITY;
    let maxFare = Number.NEGATIVE_INFINITY;
    const weightedFares: WeightedValue[] = [];
    let throughput = 0;
    rows.forEach((row: HubMarketPowerRow) => {
      const passengers = Number(row.total_passengers ?? 0);
      const distance = Number(row.avg_distance_weighted ?? 0);
      const fare = Number(row.avg_fare_weighted ?? 0);
      if (!Number.isFinite(passengers) || passengers <= 0) {
        return;
      }
      totalPassengers += passengers;
      if (Number.isFinite(fare) && fare > 0) {
        fareXPassengers += passengers * fare;
        minFare = Math.min(minFare, fare);
        maxFare = Math.max(maxFare, fare);
        weightedFares.push({ value: fare, weight: passengers });
      }
      if (Number.isFinite(distance) && distance > 0) {
        throughput += passengers * distance;
      }
    });
    return {
      period: dataset.period,
      totalPassengers,
      avgFare: totalPassengers > 0 ? fareXPassengers / totalPassengers : 0,
      minFare: Number.isFinite(minFare) ? minFare : 0,
      maxFare: Number.isFinite(maxFare) ? maxFare : 0,
      medianFare: computeWeightedMedian(weightedFares),
      avgHhi: computeAverageHubHhi(rows),
      marketSharePct: computeTopCarrierSharePctForHub(rows),
      throughput,
    };
  });
}

export function pickRoutePrice(point: RouteComparativePoint, priceMetric: RoutePriceMetric): number {
  switch (priceMetric) {
    case "max":
      return point.maxFare;
    case "min":
      return point.minFare;
    case "median":
      return point.medianFare;
    case "avg":
    default:
      return point.avgFare;
  }
}

export function pickHubPrice(point: HubComparativePoint, priceMetric: RoutePriceMetric): number {
  switch (priceMetric) {
    case "max":
      return point.maxFare;
    case "min":
      return point.minFare;
    case "median":
      return point.medianFare;
    case "avg":
    default:
      return point.avgFare;
  }
}

export function toTrendPoints<T extends { period: string }>(
  rows: T[],
  readValue: (row: T) => number,
): TrendPoint[] {
  return rows.map((row) => ({
    label: row.period,
    value: readValue(row),
  }));
}

export interface TrendDelta {
  start: number;
  end: number;
  absolute: number;
  pct: number | null;
  startLabel: string | null;
  endLabel: string | null;
}

function resolveBasePointIndex(points: TrendPoint[], basePeriod?: string): number {
  if (points.length === 0) {
    return 0;
  }
  if (!basePeriod) {
    return 0;
  }
  const index = points.findIndex((point) => point.label === basePeriod);
  return index >= 0 ? index : 0;
}

function resolveComparisonPointIndex(points: TrendPoint[], baseIndex: number): number {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    if (index !== baseIndex) {
      return index;
    }
  }
  return baseIndex;
}

export function computeTrendDelta(points: TrendPoint[], basePeriod?: string): TrendDelta {
  if (points.length === 0) {
    return { start: 0, end: 0, absolute: 0, pct: null, startLabel: null, endLabel: null };
  }
  const baseIndex = resolveBasePointIndex(points, basePeriod);
  const comparisonIndex = resolveComparisonPointIndex(points, baseIndex);
  const startPoint = points[baseIndex];
  const endPoint = points[comparisonIndex];
  const start = Number(startPoint?.value ?? 0);
  const end = Number(endPoint?.value ?? 0);
  const absolute = end - start;
  const pct = Math.abs(start) > 0 ? (absolute / start) * 100 : null;
  return {
    start,
    end,
    absolute,
    pct,
    startLabel: startPoint?.label ?? null,
    endLabel: endPoint?.label ?? null,
  };
}

export interface ComparativeChangeRow {
  label: string;
  value: number;
  start: number;
  end: number;
  change: number;
}

export interface ComparativeChangeSplit {
  increases: ComparativeChangeRow[];
  decreases: ComparativeChangeRow[];
}

function splitChangeRows(
  rows: Array<{ label: string; start: number; end: number; change: number }>,
  limit: number,
): ComparativeChangeSplit {
  const increases = rows
    .filter((row) => row.change > 0)
    .sort((a, b) => b.change - a.change)
    .slice(0, limit)
    .map((row) => ({
      label: row.label,
      value: row.change,
      start: row.start,
      end: row.end,
      change: row.change,
    }));

  const decreases = rows
    .filter((row) => row.change < 0)
    .sort((a, b) => a.change - b.change)
    .slice(0, limit)
    .map((row) => ({
      label: row.label,
      value: Math.abs(row.change),
      start: row.start,
      end: row.end,
      change: row.change,
    }));

  return { increases, decreases };
}

function aggregateRouteFareByMarket(rows: RouteMarketPowerRow[]): Map<string, { fareXPass: number; passengers: number }> {
  const map = new Map<string, { fareXPass: number; passengers: number }>();
  rows.forEach((row) => {
    const route = `${row.Origin}-${row.Dest}`;
    const passengers = Number(row.total_passengers ?? 0);
    const fare = Number(row.avg_fare_weighted ?? 0);
    if (!route || !isFinitePositive(passengers) || !isFinitePositive(fare)) {
      return;
    }
    const existing = map.get(route) ?? { fareXPass: 0, passengers: 0 };
    existing.fareXPass += fare * passengers;
    existing.passengers += passengers;
    map.set(route, existing);
  });
  return map;
}

function aggregateCarrierSharePct(rows: RouteMarketPowerRow[]): Map<string, number> {
  const passengersByCarrier = new Map<string, number>();
  let totalPassengers = 0;
  rows.forEach((row) => {
    const carrier = String(row.Carrier ?? "").trim().toUpperCase();
    const passengers = Number(row.total_passengers ?? 0);
    if (!carrier || !isFinitePositive(passengers)) {
      return;
    }
    passengersByCarrier.set(carrier, (passengersByCarrier.get(carrier) ?? 0) + passengers);
    totalPassengers += passengers;
  });
  const shareByCarrier = new Map<string, number>();
  passengersByCarrier.forEach((passengers, carrier) => {
    shareByCarrier.set(carrier, totalPassengers > 0 ? (passengers / totalPassengers) * 100 : 0);
  });
  return shareByCarrier;
}

function aggregateHubMetric(
  rows: HubMarketPowerRow[],
  readValue: (row: HubMarketPowerRow) => number,
): Map<string, number> {
  const map = new Map<string, number>();
  rows.forEach((row) => {
    const hubCode = normalizeAirportCode(row.Origin);
    const hub = getAirportDisplayName(hubCode);
    const value = readValue(row);
    if (!hub || !Number.isFinite(value)) {
      return;
    }
    map.set(hub, (map.get(hub) ?? 0) + value);
  });
  return map;
}

export function buildRouteMarketFareChange(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
  limit = 10,
  basePeriod = "",
  minPassengersPerPeriod = 0,
): ComparativeChangeSplit {
  const sorted = sortDatasetsByPeriod(datasets);
  if (sorted.length < 2) {
    return { increases: [], decreases: [] };
  }
  const baseIndex = sorted.findIndex((dataset) => dataset.period === basePeriod);
  const safeBaseIndex = baseIndex >= 0 ? baseIndex : 0;
  const baseDataset = sorted[safeBaseIndex];
  const compareDataset = sorted[sorted.length - 1].period !== baseDataset.period
    ? sorted[sorted.length - 1]
    : sorted[Math.max(0, safeBaseIndex - 1)];
  const baseRows = applyRouteFilters(baseDataset.routeRows, routeFilters);
  const compareRows = applyRouteFilters(compareDataset.routeRows, routeFilters);

  const baseMap = aggregateRouteFareByMarket(baseRows);
  const compareMap = aggregateRouteFareByMarket(compareRows);
  const keys = new Set([...baseMap.keys(), ...compareMap.keys()]);
  const rows: Array<{ label: string; start: number; end: number; change: number }> = [];

  keys.forEach((key) => {
    const base = baseMap.get(key);
    const compare = compareMap.get(key);
    const startPassengers = base?.passengers ?? 0;
    const endPassengers = compare?.passengers ?? 0;
    if (startPassengers < minPassengersPerPeriod || endPassengers < minPassengersPerPeriod) {
      return;
    }
    const start = base && base.passengers > 0 ? base.fareXPass / base.passengers : 0;
    const end = compare && compare.passengers > 0 ? compare.fareXPass / compare.passengers : 0;
    rows.push({ label: key, start, end, change: end - start });
  });

  return splitChangeRows(rows, limit);
}

export function buildRouteCarrierShareShift(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
  limit = 10,
  basePeriod = "",
): ComparativeChangeSplit {
  const sorted = sortDatasetsByPeriod(datasets);
  if (sorted.length < 2) {
    return { increases: [], decreases: [] };
  }
  const baseIndex = sorted.findIndex((dataset) => dataset.period === basePeriod);
  const safeBaseIndex = baseIndex >= 0 ? baseIndex : 0;
  const baseDataset = sorted[safeBaseIndex];
  const compareDataset = sorted[sorted.length - 1].period !== baseDataset.period
    ? sorted[sorted.length - 1]
    : sorted[Math.max(0, safeBaseIndex - 1)];
  const baseRows = applyRouteFilters(baseDataset.routeRows, routeFilters);
  const compareRows = applyRouteFilters(compareDataset.routeRows, routeFilters);
  const baseMap = aggregateCarrierSharePct(baseRows);
  const compareMap = aggregateCarrierSharePct(compareRows);
  const carriers = new Set([...baseMap.keys(), ...compareMap.keys()]);
  const rows: Array<{ label: string; start: number; end: number; change: number }> = [];

  carriers.forEach((carrier) => {
    const start = baseMap.get(carrier) ?? 0;
    const end = compareMap.get(carrier) ?? 0;
    rows.push({ label: carrier, start, end, change: end - start });
  });

  return splitChangeRows(rows, limit);
}

export function buildHubPassengerChange(
  datasets: DatasetRecord[],
  carrierFilter: string,
  hubOrigin = "",
  limit = 10,
  basePeriod = "",
  minPassengersPerPeriod = 0,
): ComparativeChangeSplit {
  const sorted = sortDatasetsByPeriod(datasets);
  if (sorted.length < 2) {
    return { increases: [], decreases: [] };
  }
  const filterOrigin = hubOrigin.trim().toUpperCase();
  const baseIndex = sorted.findIndex((dataset) => dataset.period === basePeriod);
  const safeBaseIndex = baseIndex >= 0 ? baseIndex : 0;
  const baseDataset = sorted[safeBaseIndex];
  const compareDataset = sorted[sorted.length - 1].period !== baseDataset.period
    ? sorted[sorted.length - 1]
    : sorted[Math.max(0, safeBaseIndex - 1)];
  const baseRowsRaw = applyHubFilters(baseDataset.hubRows, { carrier: carrierFilter });
  const compareRowsRaw = applyHubFilters(compareDataset.hubRows, { carrier: carrierFilter });
  const baseRows = filterOrigin
    ? baseRowsRaw.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === filterOrigin)
    : baseRowsRaw;
  const compareRows = filterOrigin
    ? compareRowsRaw.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === filterOrigin)
    : compareRowsRaw;

  const baseMap = aggregateHubMetric(baseRows, (row) => Number(row.total_passengers ?? 0));
  const compareMap = aggregateHubMetric(compareRows, (row) => Number(row.total_passengers ?? 0));
  const hubs = new Set([...baseMap.keys(), ...compareMap.keys()]);
  const rows: Array<{ label: string; start: number; end: number; change: number }> = [];

  hubs.forEach((hub) => {
    const start = baseMap.get(hub) ?? 0;
    const end = compareMap.get(hub) ?? 0;
    if (start < minPassengersPerPeriod || end < minPassengersPerPeriod) {
      return;
    }
    rows.push({ label: hub, start, end, change: end - start });
  });

  return splitChangeRows(rows, limit);
}

export function buildHubFareChange(
  datasets: DatasetRecord[],
  carrierFilter: string,
  hubOrigin = "",
  limit = 10,
  basePeriod = "",
  minPassengersPerPeriod = 0,
): ComparativeChangeSplit {
  const sorted = sortDatasetsByPeriod(datasets);
  if (sorted.length < 2) {
    return { increases: [], decreases: [] };
  }
  const filterOrigin = hubOrigin.trim().toUpperCase();
  const baseIndex = sorted.findIndex((dataset) => dataset.period === basePeriod);
  const safeBaseIndex = baseIndex >= 0 ? baseIndex : 0;
  const baseDataset = sorted[safeBaseIndex];
  const compareDataset = sorted[sorted.length - 1].period !== baseDataset.period
    ? sorted[sorted.length - 1]
    : sorted[Math.max(0, safeBaseIndex - 1)];
  const baseRowsRaw = applyHubFilters(baseDataset.hubRows, { carrier: carrierFilter });
  const compareRowsRaw = applyHubFilters(compareDataset.hubRows, { carrier: carrierFilter });
  const baseRows = filterOrigin
    ? baseRowsRaw.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === filterOrigin)
    : baseRowsRaw;
  const compareRows = filterOrigin
    ? compareRowsRaw.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === filterOrigin)
    : compareRowsRaw;

  const baseFareMap = new Map<string, { fareXPass: number; passengers: number }>();
  baseRows.forEach((row) => {
    const hubCode = normalizeAirportCode(row.Origin);
    const hub = getAirportDisplayName(hubCode);
    const passengers = Number(row.total_passengers ?? 0);
    const fare = Number(row.avg_fare_weighted ?? 0);
    if (!hub || !isFinitePositive(passengers) || !isFinitePositive(fare)) {
      return;
    }
    const existing = baseFareMap.get(hub) ?? { fareXPass: 0, passengers: 0 };
    existing.fareXPass += fare * passengers;
    existing.passengers += passengers;
    baseFareMap.set(hub, existing);
  });

  const compareFareMap = new Map<string, { fareXPass: number; passengers: number }>();
  compareRows.forEach((row) => {
    const hubCode = normalizeAirportCode(row.Origin);
    const hub = getAirportDisplayName(hubCode);
    const passengers = Number(row.total_passengers ?? 0);
    const fare = Number(row.avg_fare_weighted ?? 0);
    if (!hub || !isFinitePositive(passengers) || !isFinitePositive(fare)) {
      return;
    }
    const existing = compareFareMap.get(hub) ?? { fareXPass: 0, passengers: 0 };
    existing.fareXPass += fare * passengers;
    existing.passengers += passengers;
    compareFareMap.set(hub, existing);
  });

  const hubs = new Set([...baseFareMap.keys(), ...compareFareMap.keys()]);
  const rows: Array<{ label: string; start: number; end: number; change: number }> = [];

  hubs.forEach((hub) => {
    const base = baseFareMap.get(hub);
    const compare = compareFareMap.get(hub);
    const startPassengers = base?.passengers ?? 0;
    const endPassengers = compare?.passengers ?? 0;
    if (startPassengers < minPassengersPerPeriod || endPassengers < minPassengersPerPeriod) {
      return;
    }
    const start = base && base.passengers > 0 ? base.fareXPass / base.passengers : 0;
    const end = compare && compare.passengers > 0 ? compare.fareXPass / compare.passengers : 0;
    rows.push({ label: hub, start, end, change: end - start });
  });

  return splitChangeRows(rows, limit);
}

export interface PeriodValueChangeRow {
  period: string;
  value: number;
  changeAbs: number | null;
  changePct: number | null;
}

export function buildPeriodValueChangeRows(points: TrendPoint[], basePeriod?: string): PeriodValueChangeRow[] {
  if (points.length === 0) {
    return [];
  }
  const baseIndex = resolveBasePointIndex(points, basePeriod);
  const basePoint = points[baseIndex];
  return points.map((point) => {
    if (point.label === basePoint.label) {
      return {
        period: point.label,
        value: point.value,
        changeAbs: null,
        changePct: null,
      };
    }
    const changeAbs = point.value - basePoint.value;
    const changePct = Math.abs(basePoint.value) > 0 ? (changeAbs / basePoint.value) * 100 : null;
    return {
      period: point.label,
      value: point.value,
      changeAbs,
      changePct,
    };
  });
}

interface FareWeightedPoint {
  fare: number;
  passengers: number;
}

interface FarePeriodRows {
  period: string;
  points: FareWeightedPoint[];
}

export interface FareFrequencyBandSeries {
  band: string;
  points: TrendPoint[];
}

export interface RouteCarrierFareTrend {
  carrier: string;
  totalPassengers: number;
  periodsWithData: number;
  avgFareTrend: TrendPoint[];
  medianFareTrend: TrendPoint[];
  iqrTrend: TrendPoint[];
}

function buildFareFrequencyBandsByPeriod(periodRows: FarePeriodRows[]): FareFrequencyBandSeries[] {
  const pooled = periodRows
    .flatMap((row) => row.points)
    .filter((point) => isFinitePositive(point.fare) && isFinitePositive(point.passengers));
  if (pooled.length === 0) {
    return [];
  }

  let minFare = Number.POSITIVE_INFINITY;
  let maxFare = Number.NEGATIVE_INFINITY;
  pooled.forEach((point) => {
    minFare = Math.min(minFare, point.fare);
    maxFare = Math.max(maxFare, point.fare);
  });
  if (!Number.isFinite(minFare) || !Number.isFinite(maxFare)) {
    return [];
  }

  const span = Math.max(maxFare - minFare, 1);
  const bandCount = 4;
  const bandSize = span / bandCount;
  const bands = Array.from({ length: bandCount }, (_, index) => {
    const start = minFare + index * bandSize;
    const end = index === bandCount - 1 ? maxFare : start + bandSize;
    const label = index === bandCount - 1
      ? `$${Math.round(start)}+`
      : `$${Math.round(start)}-$${Math.round(end)}`;
    return { start, end, label, index };
  });

  return bands.map((band) => ({
    band: band.label,
    points: periodRows.map((row) => {
      let total = 0;
      let inBand = 0;
      row.points.forEach((point) => {
        total += point.passengers;
        const isLast = band.index === bandCount - 1;
        const inRange = isLast
          ? point.fare >= band.start
          : point.fare >= band.start && point.fare < band.end;
        if (inRange) {
          inBand += point.passengers;
        }
      });
      return {
        label: row.period,
        value: total > 0 ? (inBand / total) * 100 : 0,
      };
    }),
  }));
}

export function buildRouteFareFrequencyBands(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
): FareFrequencyBandSeries[] {
  const periodRows: FarePeriodRows[] = sortDatasetsByPeriod(datasets).map((dataset) => {
    const rows = applyRouteFilters(dataset.routeRows, routeFilters);
    return {
      period: dataset.period,
      points: rows
        .map((row) => ({
          fare: Number(row.avg_fare_weighted ?? 0),
          passengers: Number(row.total_passengers ?? 0),
        }))
        .filter((point) => isFinitePositive(point.fare) && isFinitePositive(point.passengers)),
    };
  });
  return buildFareFrequencyBandsByPeriod(periodRows);
}

function computeWeightedQuantile(items: WeightedValue[], quantile: number): number {
  if (items.length === 0) {
    return 0;
  }
  const sorted = [...items].sort((a, b) => a.value - b.value);
  const totalWeight = sorted.reduce((sum, item) => sum + item.weight, 0);
  if (!isFinitePositive(totalWeight)) {
    const index = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * quantile)));
    return sorted[index]?.value ?? 0;
  }
  const target = totalWeight * Math.max(0, Math.min(1, quantile));
  let running = 0;
  for (const item of sorted) {
    running += item.weight;
    if (running >= target) {
      return item.value;
    }
  }
  return sorted[sorted.length - 1]?.value ?? 0;
}

export function buildRouteCarrierFareTrends(
  datasets: DatasetRecord[],
  routeFilters: RouteFilters,
): RouteCarrierFareTrend[] {
  const sorted = sortDatasetsByPeriod(datasets);
  if (sorted.length === 0) {
    return [];
  }

  const pointsByPeriod = new Map<string, Map<string, WeightedValue[]>>();

  sorted.forEach((dataset) => {
    const rows = applyRouteFilters(Array.isArray(dataset.routeRows) ? dataset.routeRows : [], routeFilters);
    const byCarrier = new Map<string, WeightedValue[]>();
    rows.forEach((row) => {
      const carrier = String(row.Carrier ?? "").trim().toUpperCase();
      const passengers = Number(row.total_passengers ?? 0);
      const fare = Number(row.avg_fare_weighted ?? 0);
      if (!carrier || !isFinitePositive(passengers) || !isFinitePositive(fare)) {
        return;
      }
      const points = byCarrier.get(carrier) ?? [];
      points.push({ value: fare, weight: passengers });
      byCarrier.set(carrier, points);
    });
    pointsByPeriod.set(dataset.period, byCarrier);
  });
  const carrierSet = new Set<string>();
  pointsByPeriod.forEach((carrierMap) => {
    carrierMap.forEach((_, carrier) => carrierSet.add(carrier));
  });

  const series: RouteCarrierFareTrend[] = [];
  carrierSet.forEach((carrier) => {
    const avgFareTrend: TrendPoint[] = [];
    const medianFareTrend: TrendPoint[] = [];
    const iqrTrend: TrendPoint[] = [];
    let totalPassengers = 0;
    let periodsWithData = 0;

    sorted.forEach((dataset) => {
      const carrierPoints = pointsByPeriod.get(dataset.period)?.get(carrier) ?? [];
      const passengers = carrierPoints.reduce((sum, point) => sum + point.weight, 0);
      const weightedFareXPassengers = carrierPoints.reduce((sum, point) => sum + (point.value * point.weight), 0);
      const avgFare = passengers > 0 ? weightedFareXPassengers / passengers : 0;
      const medianFare = computeWeightedMedian(carrierPoints);
      const q1 = computeWeightedQuantile(carrierPoints, 0.25);
      const q3 = computeWeightedQuantile(carrierPoints, 0.75);
      const iqr = Math.max(q3 - q1, 0);
      avgFareTrend.push({ label: dataset.period, value: avgFare });
      medianFareTrend.push({ label: dataset.period, value: medianFare });
      iqrTrend.push({ label: dataset.period, value: iqr });
      totalPassengers += passengers;
      if (passengers > 0) {
        periodsWithData += 1;
      }
    });

    if (periodsWithData === 0) {
      return;
    }

    series.push({
      carrier,
      totalPassengers,
      periodsWithData,
      avgFareTrend,
      medianFareTrend,
      iqrTrend,
    });
  });

  return series.sort((a, b) => {
    if (b.totalPassengers !== a.totalPassengers) {
      return b.totalPassengers - a.totalPassengers;
    }
    return a.carrier.localeCompare(b.carrier);
  });
}

export function buildHubFareFrequencyBands(
  datasets: DatasetRecord[],
  carrierFilter: string,
  hubOrigin = "",
): FareFrequencyBandSeries[] {
  const filterOrigin = hubOrigin.trim().toUpperCase();
  const periodRows: FarePeriodRows[] = sortDatasetsByPeriod(datasets).map((dataset) => {
    const rows = applyHubFilters(dataset.hubRows, { carrier: carrierFilter });
    const scoped = filterOrigin
      ? rows.filter((row) => String(row.Origin ?? "").trim().toUpperCase() === filterOrigin)
      : rows;
    return {
      period: dataset.period,
      points: scoped
        .map((row) => ({
          fare: Number(row.avg_fare_weighted ?? 0),
          passengers: Number(row.total_passengers ?? 0),
        }))
        .filter((point) => isFinitePositive(point.fare) && isFinitePositive(point.passengers)),
    };
  });
  return buildFareFrequencyBandsByPeriod(periodRows);
}

export function buildFareFrequencyBandShift(
  series: FareFrequencyBandSeries[],
  limit = 4,
  basePeriod = "",
): ComparativeChangeSplit {
  const rows = series.map((band) => {
    const baseIndex = resolveBasePointIndex(band.points, basePeriod);
    const compareIndex = resolveComparisonPointIndex(band.points, baseIndex);
    const start = band.points[baseIndex]?.value ?? 0;
    const end = band.points[compareIndex]?.value ?? 0;
    return {
      label: band.band,
      start,
      end,
      change: end - start,
    };
  });
  return splitChangeRows(rows, limit);
}

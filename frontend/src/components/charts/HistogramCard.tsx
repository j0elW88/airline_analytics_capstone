/**
 * @file src/components/charts/HistogramCard.tsx
 * @description Reusable histogram card for fare distributions and other numeric value buckets.
 */

import { Card } from "../ui/Card";
import { formatNumber } from "../../utils/format";

interface HistogramProps {
  title: string;
  subtitle?: string;
  values: number[];
  points?: HistogramPoint[];
  buckets?: HistogramBucketInput[];
  bucketCount?: number;
  color?: string;
  maxCount?: number;
}

interface ContributorStat {
  label: string;
  weight: number;
  weightedFare: number;
  avgFare: number;
}

interface Bucket {
  label: string;
  count: number;
  fareStart?: number;
  fareEnd?: number;
  contributors: ContributorStat[];
  totalContributorWeight: number;
  tooltip?: string;
}

interface HistogramPoint {
  value: number;
  label?: string;
  weight?: number;
}

interface HistogramBucketInput {
  label: string;
  count: number;
  tooltip?: string;
}

function formatCurrencyTick(value: number): string {
  if (!Number.isFinite(value)) {
    return "$0";
  }
  const abs = Math.abs(value);
  if (abs >= 1000) {
    const compact = Math.round((value / 1000) * 10) / 10;
    const text = Number.isInteger(compact) ? compact.toFixed(0) : compact.toFixed(1);
    return `$${text}k`;
  }
  return `$${formatNumber(value)}`;
}

function computeWeightedQuantile(points: Array<{ value: number; weight: number }>, q: number): number {
  if (points.length === 0) {
    return Number.NaN;
  }
  const clampedQ = Math.min(Math.max(q, 0), 1);
  const totalWeight = points.reduce((sum, point) => sum + point.weight, 0);
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
    return points[Math.floor((points.length - 1) * clampedQ)].value;
  }
  const target = totalWeight * clampedQ;
  let running = 0;
  for (const point of points) {
    running += point.weight;
    if (running >= target) {
      return point.value;
    }
  }
  return points[points.length - 1].value;
}

function resolveHistogramDomain(points: HistogramPoint[]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    if (point.value < min) {
      min = point.value;
    }
    if (point.value > max) {
      max = point.value;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: Number.NaN, max: Number.NaN };
  }
  if (min === max || points.length < 30) {
    return { min, max };
  }

  // Focus the default histogram on the central fare mass and reduce outlier stretch.
  const weightedPoints = points
    .map((point) => ({ value: point.value, weight: Math.max(point.weight ?? 1, 0) }))
    .sort((a, b) => a.value - b.value);
  const p05 = computeWeightedQuantile(weightedPoints, 0.05);
  const p95 = computeWeightedQuantile(weightedPoints, 0.95);
  if (!Number.isFinite(p05) || !Number.isFinite(p95) || p95 <= p05) {
    return { min, max };
  }
  return { min: p05, max: p95 };
}

function buildHistogram(values: number[], bucketCount: number, points?: HistogramPoint[]): Bucket[] {
  const normalizedPoints: HistogramPoint[] = (points ?? values.map((value): HistogramPoint => ({ value })))
    .filter((point) => Number.isFinite(point.value));
  if (normalizedPoints.length === 0) {
    return [];
  }

  const { min, max } = resolveHistogramDomain(normalizedPoints);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [];
  }
  if (min === max) {
    const byContributor = new Map<string, { weight: number; weightedFare: number }>();
    normalizedPoints.forEach((point) => {
      if (!point.label) {
        return;
      }
      const existing = byContributor.get(point.label);
      const weight = point.weight ?? 1;
      if (existing) {
        existing.weight += weight;
        existing.weightedFare += point.value * weight;
      } else {
        byContributor.set(point.label, { weight, weightedFare: point.value * weight });
      }
    });
    const contributors = Array.from(byContributor.entries())
      .map(([label, stats]) => ({
        label,
        weight: stats.weight,
        weightedFare: stats.weightedFare,
        avgFare: stats.weight > 0 ? stats.weightedFare / stats.weight : min,
      }))
      .sort((a, b) => b.weight - a.weight);
    const totalContributorWeight = contributors.reduce((sum, item) => sum + item.weight, 0);
    const totalWeight = normalizedPoints.reduce((sum, point) => sum + (point.weight ?? 1), 0);
    return [{
      label: `${formatCurrencyTick(min)}`,
      count: totalWeight,
      fareStart: min,
      fareEnd: max,
      contributors,
      totalContributorWeight,
    }];
  }

  const size = (max - min) / bucketCount;
  const buckets: Bucket[] = Array.from({ length: bucketCount }, (_, index) => {
    const start = min + size * index;
    const end = start + size;
    return {
      label: `${formatCurrencyTick(start)}-${formatCurrencyTick(end)}`,
      count: 0,
      fareStart: start,
      fareEnd: end,
      contributors: [],
      totalContributorWeight: 0,
    };
  });
  const contributorMaps = Array.from(
    { length: bucketCount },
    () => new Map<string, { weight: number; weightedFare: number }>(),
  );

  normalizedPoints.forEach((point) => {
    const raw = Math.floor((point.value - min) / size);
    const bucketIndex = Math.min(Math.max(raw, 0), bucketCount - 1);
    const weight = point.weight ?? 1;
    buckets[bucketIndex].count += weight;
    if (point.label) {
      const map = contributorMaps[bucketIndex];
      const existing = map.get(point.label);
      if (existing) {
        existing.weight += weight;
        existing.weightedFare += point.value * weight;
      } else {
        map.set(point.label, { weight, weightedFare: point.value * weight });
      }
    }
  });

  buckets.forEach((bucket, index) => {
    const contributors = Array.from(contributorMaps[index].entries())
      .map(([label, stats]) => ({
        label,
        weight: stats.weight,
        weightedFare: stats.weightedFare,
        avgFare: stats.weight > 0 ? stats.weightedFare / stats.weight : bucket.fareStart ?? 0,
      }))
      .sort((a, b) => b.weight - a.weight);
    bucket.contributors = contributors;
    bucket.totalContributorWeight = contributors.reduce((sum, item) => sum + item.weight, 0);
  });

  return buckets;
}

function formatPercentShare(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0%";
  }
  if (value < 1) {
    return "<1%";
  }
  return `${value.toFixed(1)}%`;
}

function describeFarePosition(avgFare: number, bucket: Bucket): string {
  if (!Number.isFinite(avgFare) || !Number.isFinite(bucket.fareStart) || !Number.isFinite(bucket.fareEnd)) {
    return "";
  }
  const range = (bucket.fareEnd ?? 0) - (bucket.fareStart ?? 0);
  if (range <= 0) {
    return "at this fare";
  }
  const ratio = (avgFare - (bucket.fareStart ?? 0)) / range;
  if (ratio < 0.34) {
    return "near lower end";
  }
  if (ratio > 0.66) {
    return "near upper end";
  }
  return "near midpoint";
}

function bucketHoverText(bucket: Bucket): string {
  if (bucket.tooltip) {
    return bucket.tooltip;
  }
  const countText = formatNumber(bucket.count);
  if (bucket.contributors.length === 0 || bucket.totalContributorWeight <= 0) {
    return `${bucket.label}: ${countText}`;
  }
  const bucketWeightedFare = bucket.contributors.reduce((sum, item) => sum + item.weightedFare, 0);
  const bucketAvgFare = bucketWeightedFare / bucket.totalContributorWeight;
  const top = bucket.contributors.slice(0, 5);
  const parts = top.map((item) => {
    const pct = (item.weight / bucket.totalContributorWeight) * 100;
    const farePosition = describeFarePosition(item.avgFare, bucket);
    return `${item.label}: ${formatPercentShare(pct)} | avg ${formatCurrencyTick(item.avgFare)} (${farePosition})`;
  });
  const remaining = bucket.contributors.length - top.length;
  const overflow = remaining > 0 ? `\n+${remaining} more carriers` : "";
  return `${bucket.label}: ${countText}\nBucket avg fare: ${formatCurrencyTick(bucketAvgFare)}\n${parts.join("\n")}${overflow}`;
}

export function HistogramCard({
  title,
  subtitle,
  values,
  points,
  buckets,
  bucketCount = 8,
  color = "var(--chart-2)",
  maxCount,
}: HistogramProps) {
  const histogram = buckets
    ? buckets.map((bucket) => ({
      label: bucket.label,
      count: bucket.count,
      contributors: [],
      totalContributorWeight: 0,
      tooltip: bucket.tooltip,
    }))
    : buildHistogram(values, bucketCount, points);
  let localMaxCount = 0;
  for (const item of histogram) {
    if (item.count > localMaxCount) {
      localMaxCount = item.count;
    }
  }
  const providedMaxCount = typeof maxCount === "number" && Number.isFinite(maxCount) && maxCount > 0
    ? maxCount
    : null;
  const resolvedMaxCount = providedMaxCount ?? localMaxCount;
  return (
    <Card title={title} subtitle={subtitle} className="chart-card histogram-card">
      {histogram.length === 0 ? (
        <p className="muted">No fare data for current selection.</p>
      ) : (
        <div className="histogram">
          {histogram.map((bucket, index) => (
            <div key={`${bucket.label}-${index}`} className="histogram__bar-wrap">
              <div className="histogram__bar-track">
                <div
                  className="histogram__bar"
                  style={{
                    height: `${resolvedMaxCount ? Math.max(Math.min((bucket.count / resolvedMaxCount) * 100, 100), 4) : 0}%`,
                    background: color,
                  }}
                  title={bucketHoverText(bucket)}
                />
              </div>
              <span className="histogram__label" title={bucket.label}>
                {bucket.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}






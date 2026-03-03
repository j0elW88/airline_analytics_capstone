/**
 * @file src/components/charts/HistogramCard.tsx
 * @description Reusable histogram card for fare distributions and other numeric value buckets.
 */

import { Card } from "../ui/Card";
import { formatNumber } from "../../utils/format";

interface HistogramProps {
  title: string;
  values: number[];
  bucketCount?: number;
}

interface Bucket {
  label: string;
  count: number;
}

function buildHistogram(values: number[], bucketCount: number): Bucket[] {
  if (values.length === 0) {
    return [];
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [];
  }
  if (min === max) {
    return [{ label: `${formatNumber(min)}`, count: values.length }];
  }

  const size = (max - min) / bucketCount;
  const buckets: Bucket[] = Array.from({ length: bucketCount }, (_, index) => {
    const start = min + size * index;
    const end = start + size;
    return {
      label: `$${formatNumber(start)}-$${formatNumber(end)}`,
      count: 0,
    };
  });

  values.forEach((value) => {
    const raw = Math.floor((value - min) / size);
    const bucketIndex = Math.min(Math.max(raw, 0), bucketCount - 1);
    buckets[bucketIndex].count += 1;
  });

  return buckets;
}

export function HistogramCard({ title, values, bucketCount = 8 }: HistogramProps) {
  const histogram = buildHistogram(values, bucketCount);
  let maxCount = 0;
  for (const item of histogram) {
    if (item.count > maxCount) {
      maxCount = item.count;
    }
  }

  return (
    <Card title={title} className="chart-card">
      {histogram.length === 0 ? (
        <p className="muted">No fare data for current selection.</p>
      ) : (
        <div className="histogram">
          {histogram.map((bucket) => (
            <div key={bucket.label} className="histogram__bar-wrap">
              <div className="histogram__bar-track">
                <div
                  className="histogram__bar"
                  style={{
                    height: `${maxCount ? Math.max((bucket.count / maxCount) * 100, 4) : 0}%`,
                  }}
                  title={`${bucket.label}: ${bucket.count}`}
                />
              </div>
              <span className="histogram__label">{bucket.label}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}






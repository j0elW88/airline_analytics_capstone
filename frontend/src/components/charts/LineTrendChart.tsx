/**
 * @file src/components/charts/LineTrendChart.tsx
 * @description Compact line chart card for period-over-period trend visualization.
 */

import type { ReactNode } from "react";
import { Card } from "../ui/Card";
import { formatNumber } from "../../utils/format";

interface TrendPoint {
  label: string;
  value: number;
}

interface LineTrendChartProps {
  title: string;
  subtitle?: string;
  points: TrendPoint[];
  color?: string;
  valueFormatter?: (value: number) => string;
  footer?: ReactNode;
}

const WIDTH = 620;
const HEIGHT = 220;
const PAD_LEFT = 32;
const PAD_RIGHT = 14;
const PAD_TOP = 14;
const PAD_BOTTOM = 34;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function LineTrendChart({
  title,
  subtitle,
  points,
  color = "var(--chart-1)",
  valueFormatter,
  footer,
}: LineTrendChartProps) {
  if (points.length === 0) {
    return (
      <Card title={title} subtitle={subtitle} className="chart-card">
        <p className="muted">No data for current selection.</p>
      </Card>
    );
  }

  const values = points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const yRange = max - min;
  const normalizedRange = yRange > 0 ? yRange : Math.max(Math.abs(max), 1);

  const chartWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
  const chartHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;

  const chartPoints = points.map((point, index) => {
    const x = points.length > 1
      ? PAD_LEFT + (index / (points.length - 1)) * chartWidth
      : PAD_LEFT + chartWidth / 2;
    const yValue = yRange > 0 ? (point.value - min) / normalizedRange : 0.5;
    const y = PAD_TOP + (1 - clamp(yValue, 0, 1)) * chartHeight;
    return { ...point, x, y };
  });

  const polyline = chartPoints.map((point) => `${point.x},${point.y}`).join(" ");
  const startValue = chartPoints[0]?.value ?? 0;
  const endValue = chartPoints[chartPoints.length - 1]?.value ?? 0;
  const amountMax = Math.max(...points.map((point) => Math.abs(point.value)), 0);
  const changes = points.slice(1).map((point, index) => ({
    label: `${points[index].label} to ${point.label}`,
    value: point.value - points[index].value,
  }));
  const changeMax = Math.max(...changes.map((point) => Math.abs(point.value)), 0);

  function formatValue(value: number): string {
    return valueFormatter ? valueFormatter(value) : formatNumber(value);
  }

  function formatSignedValue(value: number): string {
    if (!Number.isFinite(value) || value === 0) {
      return formatValue(0);
    }
    const sign = value > 0 ? "+" : "-";
    return `${sign}${formatValue(Math.abs(value))}`;
  }

  return (
    <Card
      title={title}
      subtitle={subtitle}
      className="chart-card"
      headerRight={(
        <span className="line-trend-chart__change">
          {valueFormatter ? valueFormatter(startValue) : formatNumber(startValue)}
          {" -> "}
          {valueFormatter ? valueFormatter(endValue) : formatNumber(endValue)}
        </span>
      )}
    >
      <div className="line-trend-chart">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" className="line-trend-chart__svg">
          <line
            className="line-trend-chart__axis"
            x1={PAD_LEFT}
            y1={HEIGHT - PAD_BOTTOM}
            x2={WIDTH - PAD_RIGHT}
            y2={HEIGHT - PAD_BOTTOM}
          />
          <polyline className="line-trend-chart__line" points={polyline} style={{ stroke: color }} />
          {chartPoints.map((point) => (
            <g key={point.label}>
              <circle
                className="line-trend-chart__dot"
                cx={point.x}
                cy={point.y}
                r={4}
                style={{ fill: color }}
              >
                <title>
                  {point.label}: {valueFormatter ? valueFormatter(point.value) : formatNumber(point.value)}
                </title>
              </circle>
              <text className="line-trend-chart__label" x={point.x} y={HEIGHT - 12} textAnchor="middle">
                {point.label.replace("_", " ")}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="line-trend-chart__support">
        <div className="line-trend-chart__support-chart">
          <p className="line-trend-chart__support-title">Period Amounts</p>
          <div className="line-trend-chart__support-bars">
            {points.map((point) => {
              const height = amountMax > 0 ? Math.max((Math.abs(point.value) / amountMax) * 100, 4) : 0;
              return (
                <div key={`amount-${point.label}`} className="line-trend-chart__support-item">
                  <span className="line-trend-chart__support-value">{formatValue(point.value)}</span>
                  <div className="line-trend-chart__support-track">
                    <div
                      className="line-trend-chart__support-bar"
                      style={{ height: `${height}%`, background: color }}
                    />
                  </div>
                  <span className="line-trend-chart__support-label">{point.label.replace(/_/g, " ")}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="line-trend-chart__support-chart">
          <p className="line-trend-chart__support-title">Period Change (+/-)</p>
          {changes.length === 0 ? (
            <p className="muted">Need at least two periods.</p>
          ) : (
            <div className="line-trend-chart__support-bars">
              {changes.map((point) => {
                const height = changeMax > 0 ? Math.max((Math.abs(point.value) / changeMax) * 100, 4) : 0;
                const deltaClass = point.value > 0
                  ? "line-trend-chart__support-bar--positive"
                  : point.value < 0
                    ? "line-trend-chart__support-bar--negative"
                    : "line-trend-chart__support-bar--neutral";
                return (
                  <div key={`change-${point.label}`} className="line-trend-chart__support-item">
                    <span className="line-trend-chart__support-value">{formatSignedValue(point.value)}</span>
                    <div className="line-trend-chart__support-track">
                      <div
                        className={`line-trend-chart__support-bar ${deltaClass}`}
                        style={{ height: `${height}%` }}
                      />
                    </div>
                    <span className="line-trend-chart__support-label">{point.label.replace(/_/g, " ")}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
      {footer ? <div className="line-trend-chart__footer">{footer}</div> : null}
    </Card>
  );
}

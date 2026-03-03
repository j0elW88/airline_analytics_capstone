/**
 * @file src/features/results/SelectionGuidance.tsx
 * @description Guidance text component that explains filter-driven display modes.
 */

import { Card } from "../../components/ui/Card";
import { detectSelectionMode, getSelectionModeTitle } from "./analytics";
import type { RouteFilters } from "../../types/data";

interface SelectionGuidanceProps {
  filters: RouteFilters;
}

export function SelectionGuidance({ filters }: SelectionGuidanceProps) {
  const mode = detectSelectionMode(filters);

  return (
    <Card title={`Display Mode: ${getSelectionModeTitle(mode)}`}>
      <ul className="guidance-list">
        <li>Origin + Destination: carrier count, average fare, market share, fare comparison.</li>
        <li>Carrier + Route filters: fare histogram with focused route metrics.</li>
        <li>Carrier only: top routes, revenue proxy, mileage, passenger share, estimated flights.</li>
        <li>Hub display: hub passengers, average fare, destinations served.</li>
      </ul>
    </Card>
  );
}






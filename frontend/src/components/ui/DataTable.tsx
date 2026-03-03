/**
 * @file src/components/ui/DataTable.tsx
 * @description Generic typed table renderer for analytics datasets.
 */

import type { ReactNode } from "react";

export interface DataColumn<Row> {
  key: string;
  header: string;
  render: (row: Row) => ReactNode;
}

interface DataTableProps<Row> {
  rows: Row[];
  columns: DataColumn<Row>[];
  rowKey: (row: Row, index: number) => string;
}

export function DataTable<Row>({ rows, columns, rowKey }: DataTableProps<Row>) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey(row, index)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render(row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}






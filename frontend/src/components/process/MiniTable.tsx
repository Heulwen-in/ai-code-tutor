import type { Table } from "@/lib/processContent";

export function MiniTable({ headers, rows, caption }: Table) {
  return (
    <div className="proc-table-wrap">
      <table className="proc-table">
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

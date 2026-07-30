"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ScorePoint } from "@/lib/api";

// Two-line axis tick: weekday on top, dd/mm below.
function DayTick({ x, y, payload, data }: any) {
  const point: ScorePoint | undefined = data[payload.index];
  const short = point?.date ? point.date.slice(0, 5) : ""; // dd/mm from dd/mm/yyyy
  return (
    <g transform={`translate(${x},${y})`}>
      <text textAnchor="middle" fill="#6B7280" fontSize={12} fontWeight={700} dy={14}>
        {payload.value}
      </text>
      <text textAnchor="middle" fill="#9CA3AF" fontSize={10} dy={28}>
        {short}
      </text>
    </g>
  );
}

export function ScoreChart({ data }: { data: ScorePoint[] }) {
  return (
    <div style={{ width: "100%", height: 260, marginTop: 18 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 8, left: 8, bottom: 18 }}>
          <defs>
            <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#A855F7" />
              <stop offset="100%" stopColor="#5B5BF7" />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="#EDE9FE" />
          <XAxis
            dataKey="label"
            axisLine={false}
            tickLine={false}
            interval={0}
            height={40}
            tick={<DayTick data={data} />}
          />
          <YAxis allowDecimals={false} hide />
          <Tooltip
            cursor={{ fill: "rgba(91,91,247,0.06)" }}
            contentStyle={{
              borderRadius: 10,
              border: "1px solid #EDE9FE",
              fontSize: 12,
              boxShadow: "0 8px 24px rgba(30,27,75,0.12)",
            }}
            labelStyle={{ color: "#1E1B4B", fontWeight: 700 }}
            labelFormatter={(_, p) => (p && p[0] ? (p[0].payload as ScorePoint).date : "")}
            formatter={(value) => [`${value} pts`, "Earned"]}
          />
          <Bar dataKey="score" radius={[8, 8, 0, 0]} maxBarSize={54}>
            {data.map((_, i) => (
              <Cell key={i} fill="url(#barGrad)" />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

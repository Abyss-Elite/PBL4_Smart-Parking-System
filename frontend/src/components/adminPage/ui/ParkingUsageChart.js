"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

const COLORS = ["#22c55e", "#e5e7eb"];

export default function ParkingUsageChart({ target, label = "Đang sử dụng" }) {
  const [percent, setPercent] = useState(0);

  useEffect(() => {
    let current = 0;
    const timer = setInterval(() => {
      current += 1;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      setPercent(current);
    }, 20);
    return () => clearInterval(timer);
  }, [target]);

  const data = [
    { name: "Đang sử dụng", value: percent },
    { name: "Còn trống", value: 100 - percent },
  ];

  return (
    <Card className="relative">
      <CardHeader>
        <CardTitle>Tỷ lệ sử dụng bãi đỗ</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              startAngle={90}
              endAngle={450}
              innerRadius={80}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
              isAnimationActive={true}
              animationDuration={800}
            >
              {data.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
          <p className="text-4xl font-bold text-green-600 transition-all duration-300">
            {percent}%
          </p>
          <p className="text-sm text-gray-500">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}

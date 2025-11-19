"use client";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function UsageTrendChart() {
  const data = [
    { hour: "0h", usage: 5 },
    { hour: "1h", usage: 3 },
    { hour: "2h", usage: 2 },
    { hour: "3h", usage: 1 },
    { hour: "4h", usage: 1 },
    { hour: "5h", usage: 2 },
    { hour: "6h", usage: 5 },
    { hour: "7h", usage: 12 },
    { hour: "8h", usage: 40 },
    { hour: "9h", usage: 60 },
    { hour: "10h", usage: 75 },
    { hour: "11h", usage: 80 },
    { hour: "12h", usage: 90 },
    { hour: "13h", usage: 85 },
    { hour: "14h", usage: 70 },
    { hour: "15h", usage: 60 },
    { hour: "16h", usage: 55 },
    { hour: "17h", usage: 50 },
    { hour: "18h", usage: 45 },
    { hour: "19h", usage: 35 },
    { hour: "20h", usage: 25 },
    { hour: "21h", usage: 15 },
    { hour: "22h", usage: 10 },
    { hour: "23h", usage: 5 },
  ];

  return (
    <Card className="rounded-2xl border">
      <CardHeader>
        <CardTitle>Tỷ lệ sử dụng bãi / Giờ cao điểm</CardTitle>
      </CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="usage" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

"use client";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { parkingLotAPI } from "@/api/parking-lot/parkingLotAPI";

export default function UsageTrendChart() {
  const [data, setData] = useState([]);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await parkingLotAPI.getUsageTrend();
        setData(res.data);
      } catch (err) {
        console.error("Lỗi API biểu đồ xu hướng sử dụng:", err);
      }
    };
    fetchData();
  }, []);
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

"use client";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ReportTabs } from "@/components/reports_and_analytics/ReportTabs";
import { useEffect, useState } from "react";
import { parkingLotAPI } from "@/api/parking-lot/parkingLotAPI";
import { formatDateOnly } from "@/utils/formatDateOnly";

export default function ReportStatsPage() {
  const [initialStartDate, setInitialStartDate] = useState();
  useEffect(() => {
    const fetchData = async () => {
      const res = await parkingLotAPI.getInitialStartDate();
      const raw = formatDateOnly(res.data.firstActiveDate);
      if (!raw) return;

      const date = new Date(raw).toISOString().slice(0, 10);
      setInitialStartDate(date);
      console.log("date", date);
    };

    fetchData();
  }, []);

  return (
    <div className="space-y-6 p-6">
      <Card className="rounded-2xl shadow-md">
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Báo cáo - Thống kê</CardTitle>
        </CardHeader>
        <CardContent>
          <ReportTabs initialStartDate={initialStartDate} />
        </CardContent>
      </Card>
    </div>
  );
}

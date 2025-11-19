"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ReportWeek } from "./ReportWeek";
import { ReportMonth } from "./ReportMonth";
import UsageTrendChart from "./UsageTrendChart";

export function ReportTabs({ initialStartDate }) {
  return (
    <Tabs defaultValue="week" className="w-full">
      <TabsList className="bg-muted grid w-full grid-cols-3 rounded-xl p-1">
        <TabsTrigger value="week" className="rounded-xl">
          Theo tuần
        </TabsTrigger>
        <TabsTrigger value="month" className="rounded-xl">
          Theo tháng
        </TabsTrigger>
        <TabsTrigger value="trend" className="rounded-xl">
          Tỷ lệ sử dụng
        </TabsTrigger>
      </TabsList>

      <TabsContent value="week" className="mt-4">
        <ReportWeek initialStartDate={initialStartDate} />
      </TabsContent>

      <TabsContent value="month" className="mt-4">
        <ReportMonth />
      </TabsContent>

      <TabsContent value="trend" className="mt-4">
        <UsageTrendChart />
      </TabsContent>
    </Tabs>
  );
}

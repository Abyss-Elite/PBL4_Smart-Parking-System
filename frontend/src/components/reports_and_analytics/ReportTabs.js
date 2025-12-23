"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ReportWeek } from "./ReportWeek";
import { ReportMonth } from "./ReportMonth";
import UsageTrendChart from "./UsageTrendChart";

export function ReportTabs({ initialStartDate }) {
  return (
    <Tabs defaultValue="week" className="w-full">
      <TabsList className="bg-muted grid w-full grid-cols-2 rounded-xl p-1">
        <TabsTrigger value="week" className="cursor-pointer rounded-xl">
          Theo tuần
        </TabsTrigger>
        <TabsTrigger value="month" className="cursor-pointer rounded-xl">
          Theo tháng
        </TabsTrigger>
      </TabsList>

      <TabsContent value="week" className="mt-4">
        <ReportWeek initialStartDate={initialStartDate} />
      </TabsContent>

      <TabsContent value="month" className="mt-4">
        <ReportMonth initialStartDate={initialStartDate} />
      </TabsContent>
    </Tabs>
  );
}

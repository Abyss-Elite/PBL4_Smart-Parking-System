"use client";
import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeneralParking from "@/components/parking/generalParking/GeneralParking";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";

export default function ParkingLotManagementPage() {
  const [tab, setTab] = useState("reserved");

  return (
    <div className="space-y-6 p-6">
      <Card className="rounded-2xl shadow-md">
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Quản lý bãi xe</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={tab} onValueChange={setTab} className="w-full">
            <TabsList className="bg-muted mb-4 grid w-full grid-cols-2 rounded-xl p-1">
              <TabsTrigger value="reserved" className="rounded-xl">
                Bãi đặt trước
              </TabsTrigger>
              <TabsTrigger value="free" className="rounded-xl">
                Bãi tự do
              </TabsTrigger>
            </TabsList>

            <TabsContent value="reserved">
              <div>hihi</div>
            </TabsContent>
            <TabsContent value="free">
              <GeneralParking />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

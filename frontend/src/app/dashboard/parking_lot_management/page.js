"use client";
import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeneralParking from "@/components/parking/generalParking/GeneralParking";
import ReservationParking from "@/components/parking/reservation/ReservationParking";

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
              <TabsTrigger value="reserved" className="rounded-xl cursor-pointer">
                Bãi đặt trước
              </TabsTrigger>
              <TabsTrigger value="free" className="rounded-xl cursor-pointer">
                Bãi tự do
              </TabsTrigger>
            </TabsList>
            <TabsContent value="reserved">
              <ReservationParking />
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

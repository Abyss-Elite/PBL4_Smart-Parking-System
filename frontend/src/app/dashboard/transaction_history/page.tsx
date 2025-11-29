"use client";
import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import ReservedHistory from "@/components/transaction_history/ReservedHistory";
import FreeHistory from "@/components/transaction_history/FreeHistory";

export default function ParkingLotManagementPage() {
  const [tab, setTab] = useState("free");

  return (
    <div className="space-y-6 p-6">
      <Card className="rounded-2xl shadow-md">
        <CardHeader className={undefined}>
          <CardTitle className="text-xl font-semibold">Transaction history</CardTitle>
        </CardHeader>
        <CardContent className={undefined}>
          <Tabs value={tab} onValueChange={setTab} className="w-full">
            <TabsList className="bg-muted mb-4 grid w-full grid-cols-2 rounded-xl p-1">
              <TabsTrigger value="reserved" className="rounded-xl">
                Bãi đặt trước
              </TabsTrigger>
              <TabsTrigger value="free" className="rounded-xl">
                Bãi tự do
              </TabsTrigger>
            </TabsList>

            <TabsContent value="reserved" className={undefined}>
              <ReservedHistory />
            </TabsContent>
            <TabsContent value="free" className={undefined}>
              <FreeHistory />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

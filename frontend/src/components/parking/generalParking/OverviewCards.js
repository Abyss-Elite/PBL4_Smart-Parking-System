"use client";

import { Card, CardContent } from "@/components/ui/card";
export function OverviewCards({ parkingCondition }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {Array.isArray(parkingCondition) &&
        parkingCondition.map((item, index) => (
          <Card key={index} className={`${item.color} text-center`}>
            <CardContent className="py-4">
              <p className="text-sm">{item.title}</p>
              <p className="text-2xl font-bold">{item.value}</p>
            </CardContent>
          </Card>
        ))}
    </div>
  );
}

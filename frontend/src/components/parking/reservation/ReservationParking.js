"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ParkingSpot from "@/components/parking/ParkingSpot";
import { useRouter } from "next/navigation";
import PATH from "@/routes/PATH";
import React, { useEffect, useState } from "react";
import { carAPI } from "@/api/car/carAPI";

export default function ReservationParking() {
  const data = [
    { id: 1, name: "A1", status: "AVAILABLE" },
    { id: 2, name: "A2", status: "BOOKED" },
    { id: 3, name: "A3", status: "AVAILABLE" },
    { id: 4, name: "A4", status: "BOOKED" },
    { id: 5, name: "A5", status: "AVAILABLE" },
    { id: 6, name: "A6", status: "AVAILABLE" },
    { id: 7, name: "A7", status: "BOOKED" },
    { id: 8, name: "A8", status: "AVAILABLE" },
    { id: 9, name: "A9", status: "BOOKED" },
    { id: 10, name: "A10", status: "AVAILABLE" },
    { id: 11, name: "A11", status: "AVAILABLE" },
    { id: 12, name: "A12", status: "BOOKED" },
    { id: 13, name: "A13", status: "AVAILABLE" },
    { id: 14, name: "A14", status: "BOOKED" },
    { id: 15, name: "A15", status: "AVAILABLE" },
  ];
  const [allSpot, setAllSpot] = useState([])
  const router = useRouter();
  const handleClick = (id) => {
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.RESERVATION.DETAIL(id));
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await carAPI.getAllSpotReservation();
        // setAllSpot(res.data);
        setAllSpot(data); 
      } catch (error) {
        console.error("Error fetching spots:", error);
        setAllSpot(data);  
      }
    };

    fetchData();
  }, []);

  return (
    <div className="container mx-auto px-4 py-6">
      <Card className="overflow-x-auto rounded-lg border border-gray-200 shadow-lg">
        <CardHeader className="mb-4 flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">
            Reserved Parking
          </CardTitle>
        </CardHeader>

        <CardContent className="grid grid-cols-3 gap-6 md:grid-cols-5 justify-items-center">
          {allSpot.map((spot) => (
            <ParkingSpot
              key={spot.id}
              id={spot.id}
              name={spot.name}
              status={spot.status}
              onclickhanlde={() => handleClick(spot.id)} 
            />
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

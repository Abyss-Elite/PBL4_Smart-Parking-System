"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import VehicleCard from "@/components/vehicle/VehicleCard";
import VehicleFormDialog from "@/components/vehicle/VehicleFormDialog";
import carAPI from "@/api/car/carAPI";

const initialVehicles = {
  1: [
    {
      id: 1,
      licensePlate: "51A-12345",
      description: "Xe sedan",
      isOut: false,
      imageUrl: "https://live.staticflickr.com/65535/49932658111_30214a4229_b.jpg",
      registrationDate: "2025-09-22",
      ownerId: 1,
    },
    {
      id: 2,
      licensePlate: "51B-54321",
      description: "Xe máy",
      isOut: true,
      imageUrl: "https://media.vov.vn/sites/default/files/styles/large/public/2022-04/img-7657.jpg",
      registrationDate: "2025-08-15",
      ownerId: 1,
    },
  ],
  2: [
    {
      id: 3,
      licensePlate: "30C-98765",
      description: "Xe tải",
      isOut: false,
      imageUrl: null,
      registrationDate: "2025-07-01",
      ownerId: 2,
    },
  ],
  3: [
    {
      id: 4,
      licensePlate: "29D-11111",
      description: "Xe hatchback",
      isOut: false,
      imageUrl: null,
      registrationDate: "2025-06-12",
      ownerId: 3,
    },
  ],
};

export default function VehiclePage() {
  const { id: userId } = useParams();
  const [vehicles, setVehicles] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editCar, setEditCar] = useState(null);

  useEffect(() => {
    // TODO: gọi API backend thật, ví dụ:
    // fetch(`http://localhost:8083/api/car/user/${userId}`)
    //   .then((res) => res.json())
    //   .then((data) => setVehicles(data));
    // setVehicles(initialVehicles[userId] || []);
    const fetchData = async () => {
      const res = await carAPI.getCarByUserId(userId);
      setVehicles(res.data);
      console.log("data:", res.data);
    }
    fetchData();
  }, [userId]);

  // --- Thêm xe mới ---
  const handleAddCar = async (newCar) => {
    const carToAdd = { ...newCar, id: Date.now(), ownerId: parseInt(userId) };
    setVehicles((prev) => [...prev, carToAdd]);
    setDialogOpen(false);

    // TODO: gọi API POST create
    // await fetch(`http://localhost:8083/api/car`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(carToAdd),
    // });
  };

  // --- Cập nhật xe ---
  const handleEditCar = async (updatedCar) => {
    setVehicles((prev) => prev.map((v) => (v.id === updatedCar.id ? updatedCar : v)));
    setEditCar(null);
    setDialogOpen(false);

    // TODO: gọi API PUT update
    // await fetch(`http://localhost:8083/api/car/${updatedCar.id}`, {
    //   method: "PUT",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(updatedCar),
    // });
  };

  // --- Xóa xe ---
  const handleDeleteCar = async (carId) => {
    if (!confirm("Bạn có chắc muốn xóa xe này?")) return;
    setVehicles((prev) => prev.filter((v) => v.id !== carId));

    // TODO: gọi API DELETE
    // await fetch(`http://localhost:8083/api/car/${carId}`, { method: "DELETE" });
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Xe của user {userId}</h1>
        <Button
          className="bg-green-600 text-white hover:bg-green-700"
          onClick={() => {
            setEditCar(null);
            setDialogOpen(true);
          }}
        >
          Đăng ký xe mới
        </Button>
      </div>

      {vehicles.length === 0 && <p className="text-gray-500">Chưa có xe nào.</p>}

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        {vehicles.map((car) => (
          <VehicleCard
            key={car.id}
            car={car}
            onEdit={(car) => {
              setEditCar(car);
              setDialogOpen(true);
            }}
            onDelete={handleDeleteCar}
          />
        ))}
      </div>

      <VehicleFormDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onSubmit={editCar ? handleEditCar : handleAddCar}
        initialData={editCar}
      />
    </div>
  );
}

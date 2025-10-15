"use client";

import { Button } from "@/components/ui/button";

export default function VehicleCard({ car, onEdit, onDelete }) {
  return (
    <div className="flex flex-col rounded-lg border bg-white p-4 shadow transition hover:shadow-lg">
      <div className="mb-4 flex justify-center">
        <img
          src={car.imageUrl || "/placeholder-car.png"}
          alt={car.licensePlate}
          className="h-20 w-32 rounded object-cover"
        />
      </div>
      <p>
        <strong>Biển số:</strong> {car.licensePlate}
      </p>
      {car.description && (
        <p>
          <strong>Mô tả:</strong> {car.description}
        </p>
      )}
      <p>
        <strong>Ngày đăng ký:</strong> {car.registrationDate}
      </p>
      <p>
        <strong>Trạng thái:</strong>{" "}
        <span className={car.isOut ? "text-red-600" : "text-green-600"}>
          {car.isOut ? "Đang ra ngoài" : "Trong bãi"}
        </span>
      </p>
      <div className="mt-4 flex justify-end space-x-2">
        <Button size="sm" variant="outline" onClick={() => onEdit(car)}>
          Edit
        </Button>
        <Button size="sm" variant="destructive" onClick={() => onDelete(car.id)}>
          Delete
        </Button>
      </div>
    </div>
  );
}

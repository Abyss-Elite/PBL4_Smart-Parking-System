"use client";

import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

export default function VehicleFormDialog({ open, onOpenChange, onSubmit, initialData }) {
  const [car, setCar] = useState({
    licensePlate: "",
    description: "",
    imageUrl: "",
    registrationDate: new Date().toISOString().split("T")[0],
    isOut: false,
  });

  useEffect(() => {
    if (initialData) setCar(initialData);
    else
      setCar({
        licensePlate: "",
        description: "",
        imageUrl: "",
        registrationDate: new Date().toISOString().split("T")[0],
        isOut: false,
      });
  }, [initialData, open]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(car);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{initialData ? "Chỉnh sửa xe" : "Đăng ký xe mới"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="mt-2 space-y-4">
          <div className="flex flex-col space-y-1">
            <Label>Biển số</Label>
            <Input
              value={car.licensePlate}
              onChange={(e) => setCar({ ...car, licensePlate: e.target.value })}
              required
            />
          </div>

          <div className="flex flex-col space-y-1">
            <Label>Mô tả</Label>
            <Input
              value={car.description}
              onChange={(e) => setCar({ ...car, description: e.target.value })}
            />
          </div>

          <div className="flex flex-col space-y-1">
            <Label>Hình ảnh URL</Label>
            <Input
              value={car.imageUrl}
              onChange={(e) => setCar({ ...car, imageUrl: e.target.value })}
            />
          </div>

          <div className="flex flex-col space-y-1">
            <Label>Ngày đăng ký</Label>
            <Input
              type="date"
              value={car.registrationDate}
              onChange={(e) => setCar({ ...car, registrationDate: e.target.value })}
            />
          </div>

          <div className="flex justify-end space-x-2">
            <Button type="submit" className="bg-green-600 text-white hover:bg-green-700">
              Lưu
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

"use client";

import { useState } from "react";
import Image from "next/image";

export default function ProfilePage() {
  // Fake data demo (bạn thay bằng API thật)
  const [form, setForm] = useState({
    avatar: "https://images.unsplash.com/photo-1502685104226-ee32379fefbe?w=300",
    fullName: "John Doe",
    email: "john@example.com",
  });

  const [initialData] = useState(form);

  // Handle change
  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  // Save button (gọi API của bạn)
  const handleSave = () => {
    console.log("Saving data:", form);
    alert("Saved!");
  };

  // Reset button
  const handleReset = () => {
    setForm(initialData);
  };

  return (
    <div className="flex w-full justify-center py-10">
      <div className="w-[700px] rounded-2xl border bg-white p-10 shadow-sm">
        <h2 className="mb-2 text-2xl font-semibold">Profile Settings</h2>
        <p className="mb-8 text-gray-500">Manage your account settings and profile information.</p>

        {/* --- AVATAR + URL --- */}
        <div className="mb-8 flex items-start gap-6">
          <Image
            src={form.avatar}
            width={90}
            height={90}
            alt="avatar"
            className="rounded-full border"
          />

          <div className="w-full">
            <label className="mb-2 block font-medium text-gray-700">Avatar URL</label>
            <input
              type="text"
              name="avatar"
              value={form.avatar}
              onChange={handleChange}
              className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
            />
          </div>
        </div>

        {/* --- FULL NAME --- */}
        <div className="mb-6">
          <label className="mb-2 block font-medium text-gray-700">Full Name</label>
          <input
            type="text"
            name="fullName"
            value={form.fullName}
            onChange={handleChange}
            className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
          />
        </div>

        {/* --- EMAIL ADDRESS --- */}
        <div className="mb-10">
          <label className="mb-2 block font-medium text-gray-700">Email Address</label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
          />
        </div>

        {/* --- BUTTONS --- */}
        <div className="flex gap-4">
          <button
            onClick={handleSave}
            className="rounded-lg bg-black px-6 py-2 text-white hover:bg-gray-800"
          >
            Save Changes
          </button>

          <button onClick={handleReset} className="rounded-lg border px-6 py-2 hover:bg-gray-100">
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
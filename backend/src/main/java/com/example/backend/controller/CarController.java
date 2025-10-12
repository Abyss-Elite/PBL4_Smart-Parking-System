package com.example.backend.controller;
import com.example.backend.model.Car;
import com.example.backend.model.User;
import com.example.backend.service.CarService;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PathVariable;
import com.example.backend.DTO.LicensePlateRequest;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
@RestController
@RequestMapping("/api/car")
// @CrossOrigin(origins = "*")
@CrossOrigin(origins = "http://192.168.1.12:5000")


public class CarController {
    @Autowired
    private CarService carService;

    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping("")
    public List<Car> getAllCars(){
        return carService.getAllCars();
    }

    @PreAuthorize("hasAnyRole('ADMIN', 'USER')")
    @GetMapping("/{id}")
    public Car getCar(@PathVariable Long id){
        return carService.getCar(id);
    }

    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping()
    public Car createCar(@RequestBody Car car){
        return carService.createCar(car);
    }

    @PreAuthorize("hasAnyRole('ADMIN', 'USER')")
    @PutMapping("/{id}")
    public Car updateCar(@PathVariable Long id, @RequestBody Car carDetails){
        return carService.updateCar(id, carDetails);
    }


    @PreAuthorize("hasRole('ADMIN')")
    @DeleteMapping("/{id}")
    public ResponseEntity<Map<String, String>> deleteCar(@PathVariable Long id) {
        try {
            carService.deleteCar(id);
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Xóa xe thành công!"
            ));
        } catch (RuntimeException e) { 
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(
                "status", "error",
                "message", "Xe không tồn tại."
            ));
        }
    }
    @GetMapping("/byUser/{userId}")
    public List<Car> getCarsByUserId(@PathVariable Long userId){
      
        return carService.getCarsByUserId(userId);
    }

    @PostMapping("/LicensePlateNumber")
    public Car getLicensePlateNumber(@RequestBody LicensePlateRequest licensePlate){
        String plate = licensePlate.getPlate();
        if(plate == null || plate.isEmpty()) throw new RuntimeException("Fail to get license plate number");
        Car car = carService.getCarsByLicensePlateNumber(plate);
        if(car == null) throw new RuntimeException("Car not found with plate: " + plate);
        return car;
    }
}

package com.example.backend.repository;

import com.example.backend.model.Car;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
@Repository
public interface CarRepository extends JpaRepository<Car, Long> {
    // Nếu muốn tìm theo biển số:
    boolean existsByLicensePlate(String licensePlate);
    // Tìm xe theo userId
    List<Car> findByOwner_Id(Long userId);
    Car findByLicensePlate(String licensePlate);

}

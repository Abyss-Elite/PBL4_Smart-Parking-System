package com.example.backend.model;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "cars")
public class Car {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable=false, unique = true)
    private String licensePlate;

    @Column(nullable=true)
    private String description;

    @Column(nullable=false)
    private Boolean isOut = false; 

    @Column(nullable=true)
    private String imageUrl;

    @Column(nullable=false)
    private String status = "ACTIVE"; 

    @Column(nullable=false)
    private LocalDate registrationDate;

    @Column(nullable = false)
    private Boolean isDelete = false; 

    @Column(nullable = false)
    private LocalDateTime lastTime; 
    
    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User owner;

    // ===== Getter & Setter =====
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getLicensePlate() { return licensePlate; }
    public void setLicensePlate(String licensePlate) { this.licensePlate = licensePlate; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public Boolean getIsOut() { return isOut; }
    public void setIsOut(Boolean isOut) { this.isOut = isOut; }

    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public LocalDate getRegistrationDate() { return registrationDate; }
    public void setRegistrationDate(LocalDate registrationDate) { this.registrationDate = registrationDate; }

    public Boolean getIsDelete() { return isDelete; }
    public void setIsDelete(Boolean isDelete) { this.isDelete = isDelete; }

    public LocalDateTime getLastTime() { return lastTime; }
    public void setLastTime(LocalDateTime lastTime) { this.lastTime = lastTime; }

    public User getOwner() { return owner; }
    public void setOwner(User owner) { this.owner = owner; }
}

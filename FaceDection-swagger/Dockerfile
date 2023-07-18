# Use an official OpenJDK runtime as the base image
FROM openjdk:11-jdk-slim

# Set the working directory in the container
WORKDIR /app

# Copy the executable JAR file into the container
COPY target/demo-0.0.1-SNAPSHOT.jar app.jar

# Expose the port that your Spring Boot application listens on
EXPOSE 8080:8080

# Set the entrypoint command to run the application when the container starts
ENTRYPOINT ["java", "-jar", "app.jar"]

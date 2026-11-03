-- ================================================================
-- Creación de la base de datos: Asignacion_MEC
-- Proyecto Conacyt-Uninter
-- Tutor investigador: Dr. Fabio Lopez
-- Investigador en formación: Ing. Eliana Telesca
-- Versión: 1.0
-- DESBALANCEADO CDE ↔ Minga Guazú (y algunos de Hernandarias)
-- ================================================================
-- 1. Crear Base de Datos (ejecutar con privilegios)
CREATE DATABASE "Asignacion_MEC";
\c Asignacion_MEC;

-- ================================================================
-- 2. TABLAS
-- ================================================================
-- Tabla INSTITUCIONES 
CREATE TABLE instituciones (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    departamento VARCHAR(50) NOT NULL,
    localidad VARCHAR(50) NOT NULL,
    barrio VARCHAR(50)
);
-- Tabla ESTABLECIMIENTOS 
CREATE TABLE establecimientos (
    id SERIAL PRIMARY KEY,
    institucion_id INTEGER NOT NULL,
    lat DECIMAL(10, 8),
    lng DECIMAL(11, 8),
    FOREIGN KEY (institucion_id) REFERENCES instituciones(id)
);
-- Tabla CLASES
CREATE TABLE clases (
    id SERIAL PRIMARY KEY,
    grado VARCHAR(20) NOT NULL,
    turno VARCHAR(10) NOT NULL,
    capacidad INTEGER NOT NULL,
    establecimiento_id INTEGER NOT NULL,
    FOREIGN KEY (establecimiento_id) REFERENCES establecimientos(id)
);
-- Tabla ESTUDIANTES
CREATE TABLE estudiantes (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    grado VARCHAR(20) NOT NULL,
    lat DECIMAL(10, 8),
    lng DECIMAL(11, 8),
    departamento VARCHAR(50),
    localidad VARCHAR(50),
    barrio VARCHAR(50)
);
-- Tabla DOCENTES (corregido de DOCENTE)
CREATE TABLE docentes (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    grado VARCHAR(20) NOT NULL,
    lat DECIMAL(10, 8),
    lng DECIMAL(11, 8),
    departamento VARCHAR(50),
    localidad VARCHAR(50),
    barrio VARCHAR(50)
);
-- Tabla para almacenar las asignaciones
CREATE TABLE asignacion_mec (
    id SERIAL PRIMARY KEY,
    estudiante_id INTEGER NOT NULL,
    docente_id INTEGER NOT NULL,
    establecimiento_id INTEGER NOT NULL,
 institucion_id INTEGER NOT NULL,
    grado VARCHAR(20) NOT NULL,
    seccion VARCHAR(10),
    turno VARCHAR(10) NOT NULL,
    distancia DECIMAL(10, 2),
    fecha_asignacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (estudiante_id) REFERENCES estudiantes(id),
    FOREIGN KEY (docente_id) REFERENCES docentes(id),
    FOREIGN KEY (establecimiento_id) REFERENCES establecimientos(id)
);

-- ================================================================
-- 3. INSERTS DE EJEMPLO (Datos mínimos para probar el proyecto)
-- ================================================================

-- ===============================
-- INSTITUCIONES (Alto Paraná)
-- ===============================
INSERT INTO instituciones (nombre, departamento, localidad, barrio) VALUES
('Colegio Nacional Ciudad del Este', 'Alto Paraná', 'Ciudad del Este', 'Centro'),               -- id 1
('Escuela Básica Minga Guazú N° 1',  'Alto Paraná', 'Minga Guazú',     'San Miguel'),           -- id 2
('Colegio Nacional Hernandarias',     'Alto Paraná', 'Hernandarias',    'Remansito'),            -- id 3
('Escuela Básica Pdte. Franco',       'Alto Paraná', 'Presidente Franco','Maria Auxiliadora');   -- id 4

-- ===============================
-- ESTABLECIMIENTOS (con coordenadas)
-- ===============================
-- CDE (id 1)
INSERT INTO establecimientos (institucion_id, lat, lng) VALUES
(1, -25.51600000, -54.61600000),   -- id 1: CDE Centro
(1, -25.53000000, -54.64000000);   -- id 2: CDE Area Este

-- Minga Guazú (id 2)
INSERT INTO establecimientos (institucion_id, lat, lng) VALUES
(2, -25.48200000, -54.82700000),   -- id 3: MG Zona 1
(2, -25.50500000, -54.82400000);   -- id 4: MG Zona 2

-- Hernandarias (id 3)
INSERT INTO establecimientos (institucion_id, lat, lng) VALUES
(3, -25.40700000, -54.64000000);   -- id 5: Hernandarias Centro

-- Pdte. Franco (id 4)
INSERT INTO establecimientos (institucion_id, lat, lng) VALUES
(4, -25.56000000, -54.61000000);   -- id 6: Franco Centro

-- ===============================
-- CLASES (capacidad pequeña para forzar presión)
-- ===============================
INSERT INTO clases (grado, turno, capacidad, establecimiento_id) VALUES
('1ro', 'Mañana', 20, 3),  -- Minga Guazú
('2do', 'Mañana', 20, 3),
('3ro', 'Tarde',  20, 4),  -- Minga Guazú
('1ro', 'Mañana', 20, 1),  -- CDE
('2do', 'Tarde',  20, 2),  -- CDE
('3ro', 'Mañana', 20, 1),  -- CDE
('1ro', 'Mañana', 20, 5),  -- Hernandarias
('2do', 'Tarde',  20, 6),  -- Pdte. Franco
('3ro', 'Mañana', 20, 5);  -- Hernandarias

-- ===============================
-- ESTUDIANTES
-- 12 en CDE, 6 en Minga Guazú, 6 en Hernandarias
-- ===============================
-- CDE (quedarán MAL asignados a Minga Guazú)
INSERT INTO estudiantes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Juan Pérez',          '1ro', -25.51750000, -54.62050000, 'Alto Paraná','Ciudad del Este','Centro'),
('María García',        '2do', -25.52000000, -54.61800000, 'Alto Paraná','Ciudad del Este','Boquerón'),
('Carlos López',        '3ro', -25.52550000, -54.63000000, 'Alto Paraná','Ciudad del Este','San José'),
('Ana Fernández',       '1ro', -25.52800000, -54.64200000, 'Alto Paraná','Ciudad del Este','Area 1'),
('Luis González',       '2do', -25.53200000, -54.63500000, 'Alto Paraná','Ciudad del Este','Santa Ana'),
('Laura Martínez',      '3ro', -25.51500000, -54.61200000, 'Alto Paraná','Ciudad del Este','Centro'),
('Miguel Rojas',        '1ro', -25.52700000, -54.64500000, 'Alto Paraná','Ciudad del Este','Area 3'),
('Patricia Bogado',     '2do', -25.51900000, -54.62100000, 'Alto Paraná','Ciudad del Este','Centro'),
('Diego Sánchez',       '3ro', -25.52200000, -54.62800000, 'Alto Paraná','Ciudad del Este','San Lucas'),
('Cynthia Duarte',      '1ro', -25.53100000, -54.64400000, 'Alto Paraná','Ciudad del Este','Area 4'),
('Hugo Vera',           '2do', -25.51700000, -54.61400000, 'Alto Paraná','Ciudad del Este','Centro'),
('Rocío Benítez',       '3ro', -25.52350000, -54.63300000, 'Alto Paraná','Ciudad del Este','San Rafael');

-- Minga Guazú (quedarán MAL asignados a CDE)
INSERT INTO estudiantes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Sofía Romero',        '1ro', -25.48500000, -54.83000000, 'Alto Paraná','Minga Guazú','San Miguel'),
('Pedro Acuña',         '2do', -25.49000000, -54.82600000, 'Alto Paraná','Minga Guazú','San Roque'),
('Valeria Cabrera',     '3ro', -25.50000000, -54.82300000, 'Alto Paraná','Minga Guazú','San Isidro'),
('Jorge Giménez',       '1ro', -25.47500000, -54.83500000, 'Alto Paraná','Minga Guazú','San Carlos'),
('Nadia Fleitas',       '2do', -25.49700000, -54.82900000, 'Alto Paraná','Minga Guazú','San Jorge'),
('Ramón Ortiz',         '3ro', -25.50300000, -54.82200000, 'Alto Paraná','Minga Guazú','San Blas');

-- Hernandarias (también quedarán MAL a Minga/CDE)
INSERT INTO estudiantes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Elena Rivas',         '1ro', -25.40800000, -54.64100000, 'Alto Paraná','Hernandarias','Centro'),
('Pablo Ferreira',      '2do', -25.41500000, -54.63800000, 'Alto Paraná','Hernandarias','San Antonio'),
('Camila Ávalos',       '3ro', -25.42000000, -54.62000000, 'Alto Paraná','Hernandarias','San José'),
('Rodrigo Núñez',       '1ro', -25.40500000, -54.64500000, 'Alto Paraná','Hernandarias','San Juan'),
('Paola Sosa',          '2do', -25.41200000, -54.63600000, 'Alto Paraná','Hernandarias','San Roque'),
('Gabriel Gómez',       '3ro', -25.41800000, -54.62500000, 'Alto Paraná','Hernandarias','Santa Teresa');

-- ===============================
-- DOCENTES (Residencias invertidas para forzar desajuste)
-- 4 en CDE, 3 en Minga, 2 en Hernandarias
-- ===============================
-- Docentes que VIVEN en CDE 
INSERT INTO docentes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Prof. Silvia Duarte', '1ro', -25.51700000, -54.61900000, 'Alto Paraná','Ciudad del Este','Centro'),
('Prof. Ricardo Ruiz',  '2do', -25.52900000, -54.64100000, 'Alto Paraná','Ciudad del Este','Area 2'),
('Prof. Marta Vázquez', '3ro', -25.52100000, -54.62600000, 'Alto Paraná','Ciudad del Este','San José'),
('Prof. Daniel Torres', '1ro', -25.52500000, -54.63200000, 'Alto Paraná','Ciudad del Este','Boquerón');

-- Docentes que VIVEN en Minga
INSERT INTO docentes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Prof. Alicia López',  '2do', -25.49100000, -54.82800000, 'Alto Paraná','Minga Guazú','San Miguel'),
('Prof. Nelson Ríos',   '3ro', -25.49800000, -54.82150000, 'Alto Paraná','Minga Guazú','San Isidro'),
('Prof. Teresa Báez',   '1ro', -25.47650000, -54.83400000, 'Alto Paraná','Minga Guazú','San Carlos');

-- Docentes que VIVEN en Hernandarias 
INSERT INTO docentes (nombre, grado, lat, lng, departamento, localidad, barrio) VALUES
('Prof. Hugo Caballero', '2do', -25.40900000, -54.63900000, 'Alto Paraná','Hernandarias','Centro'),
('Prof. Laura Patiño',   '3ro', -25.41600000, -54.63700000, 'Alto Paraná','Hernandarias','San Antonio');

-- ===============================
-- ASIGNACIONES DESBALANCEADAS
-- Regla: alumnos de CDE → establecimientos de Minga (3 y 4)
--        alumnos de Minga → establecimientos de CDE (1 y 2)
--        alumnos de Hernandarias → también mal (3/4 o 1/2)
-- Docentes también cruzados para aumentar el desajuste
-- ===============================

-- Helper: asumimos IDs consecutivos según inserción:
-- Establecimientos: 1=CDE1, 2=CDE2, 3=MG1, 4=MG2, 5=Hern, 6=Franco
-- Docentes: 1..9 según inserción; Estudiantes: 1..24

-- CDE (12 estudiantes) → Minga (3/4) con docentes que VIVEN en CDE (1,2,3,4) dictando en Minga
INSERT INTO asignacion_mec (estudiante_id, docente_id, establecimiento_id, institucion_id, grado, seccion, turno, distancia)
VALUES
(1,  1, 3, 2, '1ro', 'A', 'Mañana', 8.0),
(2,  2, 3, 2, '2do', 'A', 'Mañana', 8.2),
(3,  3, 4, 2, '3ro', 'B', 'Tarde',  8.5),
(4,  1, 3, 2, '1ro', 'B', 'Mañana', 7.9),
(5,  2, 4, 2, '2do', 'B', 'Mañana', 8.3),
(6,  3, 4, 2, '3ro', 'A', 'Tarde',  8.6),
(7,  4, 3, 2, '1ro', 'C', 'Mañana', 8.1),
(8,  2, 3, 2, '2do', 'C', 'Mañana', 8.4),
(9,  3, 4, 2, '3ro', 'C', 'Tarde',  8.7),
(10, 1, 3, 2, '1ro', 'D', 'Mañana', 7.8),
(11, 2, 4, 2, '2do', 'D', 'Mañana', 8.5),
(12, 3, 4, 2, '3ro', 'D', 'Tarde',  8.9);

-- Minga (6 estudiantes) → CDE (1/2) con docentes que VIVEN en Minga (5,6,7) dictando en CDE
INSERT INTO asignacion_mec (estudiante_id, docente_id, establecimiento_id, institucion_id, grado, seccion, turno, distancia)
VALUES
(13,  5, 1, 1, '1ro', 'A', 'Mañana', 8.0),
(14,  5, 2, 1, '2do', 'A', 'Tarde',  8.3),
(15,  6, 1, 1, '3ro', 'A', 'Mañana', 7.9),
(16,  7, 2, 1, '1ro', 'B', 'Mañana', 8.2),
(17,  5, 1, 1, '2do', 'B', 'Tarde',  8.4),
(18,  6, 2, 1, '3ro', 'B', 'Mañana', 8.6);

-- Hernandarias (6 estudiantes) → también mal: 3 a Minga (3/4) y 3 a CDE (1/2)
-- con docentes de Hernandarias (8,9) dictando lejos
INSERT INTO asignacion_mec (estudiante_id, docente_id, establecimiento_id, institucion_id, grado, seccion, turno, distancia)
VALUES
(19,  8, 3, 2, '1ro', 'A', 'Mañana', 15.0),  -- Hern → Minga
(20,  8, 1, 1, '2do', 'A', 'Tarde',  12.1),  -- Hern → CDE
(21,  9, 4, 2, '3ro', 'A', 'Tarde',  14.8),  -- Hern → Minga
(22,  8, 2, 1, '1ro', 'B', 'Mañana', 11.9),  -- Hern → CDE
(23,  8, 3, 2, '2do', 'B', 'Mañana', 15.2),  -- Hern → Minga
(24,  9, 2, 1, '3ro', 'B', 'Mañana', 12.3);  -- Hern → CDE

-- ===============================
-- CONSULTAS DE CONTROL  
-- ===============================
-- 1) Ver cuántos estudiantes están asignados fuera de su localidad vs. localidad del establecimiento
-- Estudiante vs localidad de la institución del establecimiento asignado
-- (muestra los que NO coinciden)
SELECT a.id as asignacion_id, e.nombre as estudiante, e.localidad AS vive_en,
        i.nombre as institucion, i.localidad AS clase_en, c.grado, a.turno
FROM asignacion_mec a
JOIN estudiantes e ON e.id = a.estudiante_id
JOIN establecimientos est ON est.id = a.establecimiento_id
JOIN instituciones i ON i.id = est.institucion_id
JOIN clases c ON c.establecimiento_id = est.id AND c.grado = a.grado
WHERE e.localidad <> i.localidad
ORDER BY e.localidad, i.localidad;

-- 2) Vista rápida de presión por establecimiento (cuántos asignados por grado/turno)
SELECT est.id AS est_id, i.localidad, a.grado, a.turno, COUNT(*) AS asignados
FROM asignacion_mec a
JOIN establecimientos est ON est.id=a.establecimiento_id
JOIN instituciones i ON i.id=est.institucion_id
GROUP BY est.id, i.localidad, a.grado, a.turno
ORDER BY i.localidad, est.id, a.grado, a.turno;

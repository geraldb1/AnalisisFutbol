USE football_prediction
-- =============================================
-- Football Prediction - La Liga
-- DDL: Creación de tablas
-- =============================================

-- 1. DIM_EQUIPO
CREATE TABLE dim_equipo (
    id_equipo   INT IDENTITY(1,1) PRIMARY KEY,
    nombre      NVARCHAR(100) NOT NULL
);
GO

-- 2. DIM_FECHA
CREATE TABLE dim_fecha (
    fecha       DATE PRIMARY KEY,
    Año         INT,
    Mes         INT,
    Nombre_Mes  NVARCHAR(20),
    Dia         INT,
    Nombre_Dia  NVARCHAR(20),
    Semana      INT,
    temporada   NVARCHAR(10)
);
GO

-- 3. FACT_PARTIDO
CREATE TABLE fact_partido (
    id_partido          INT IDENTITY(1,1) PRIMARY KEY,
    Date                DATE,
    HomeTeam            NVARCHAR(100),
    AwayTeam            NVARCHAR(100),
    FTHG                INT,
    FTAG                INT,
    FTR                 NVARCHAR(1),
    HTHG                INT,
    HTAG                INT,
    HTR                 NVARCHAR(1),
    temporada           NVARCHAR(10),
    Time                NVARCHAR(10),

    -- Stats
    HS FLOAT, [AS] FLOAT, HST FLOAT, AST FLOAT,
    HF FLOAT, AF FLOAT, HC FLOAT, AC FLOAT,
    HY FLOAT, AY FLOAT, HR FLOAT, AR FLOAT,

    -- Odds Match Result
    B365H FLOAT, B365D FLOAT, B365A FLOAT,
    MaxH FLOAT, MaxD FLOAT, MaxA FLOAT,
    AvgH FLOAT, AvgD FLOAT, AvgA FLOAT,

    -- Odds Over/Under 2.5
    [B365>2.5] FLOAT, [B365<2.5] FLOAT,
    [Max>2.5] FLOAT,  [Max<2.5] FLOAT,
    [Avg>2.5] FLOAT,  [Avg<2.5] FLOAT,
    [GB>2.5] FLOAT,   [GB<2.5] FLOAT,
    [P>2.5] FLOAT,    [P<2.5] FLOAT,
    [BbMx>2.5] FLOAT, [BbAv>2.5] FLOAT,
    [BbMx<2.5] FLOAT, [BbAv<2.5] FLOAT,
    [B365C>2.5] FLOAT,[B365C<2.5] FLOAT,
    [PC>2.5] FLOAT,   [PC<2.5] FLOAT,
    [MaxC>2.5] FLOAT, [MaxC<2.5] FLOAT,
    [AvgC>2.5] FLOAT, [AvgC<2.5] FLOAT,
    [BFE>2.5] FLOAT,  [BFE<2.5] FLOAT,
    [BFEC>2.5] FLOAT, [BFEC<2.5] FLOAT,
    Bb1X2 FLOAT,
    BbMxH FLOAT, BbAvH FLOAT,
    BbMxD FLOAT, BbAvD FLOAT,
    BbMxA FLOAT, BbAvA FLOAT,
    BbOU FLOAT,

    -- FKs
    id_equipo_local     INT,
    id_equipo_visitante INT,

    CONSTRAINT fk_fecha     FOREIGN KEY (Date)                REFERENCES dim_fecha(fecha),
    CONSTRAINT fk_local     FOREIGN KEY (id_equipo_local)     REFERENCES dim_equipo(id_equipo),
    CONSTRAINT fk_visitante FOREIGN KEY (id_equipo_visitante) REFERENCES dim_equipo(id_equipo)
);
GO

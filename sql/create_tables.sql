-- Creation of product table
CREATE TABLE IF NOT EXISTS sentinel2a (
  uuid VARCHAR NOT NULL,
  product_uri VARCHAR NOT NULL,
  country VARCHAR,
  continent VARCHAR,
  b02 FLOAT,
  b03 FLOAT,
  b04 FLOAT,
  b08 FLOAT,
  season VARCHAR,
  climate VARCHAR,
  classification INT,
  capture VARCHAR,
  lat FLOAT,
  lon FLOAT,
  PRIMARY KEY (uuid)
);
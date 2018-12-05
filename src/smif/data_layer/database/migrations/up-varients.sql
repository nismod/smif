CREATE TABLE variants(
	"id" serial PRIMARY KEY,
	"variant_name" varchar,
	"description" varchar,
	"data" JSONB
);
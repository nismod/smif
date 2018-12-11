CREATE TABLE variants(
	"id" serial PRIMARY KEY,
	"name" varchar,
	"description" varchar,
	"data" JSONB
);
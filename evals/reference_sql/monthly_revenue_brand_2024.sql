SELECT
  FORMAT_DATE('%Y-%m', DATE(oi.created_at)) AS month,
  ROUND(SUM(CAST(oi.sale_price AS NUMERIC)), 2) AS revenue,
  COUNT(DISTINCT oi.user_id) AS distinct_customers
FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi
JOIN `bigquery-public-data.thelook_ecommerce.orders` AS o
  ON o.order_id = oi.order_id AND o.user_id = oi.user_id
JOIN `bigquery-public-data.thelook_ecommerce.products` AS p
  ON p.id = oi.product_id
JOIN `bigquery-public-data.thelook_ecommerce.users` AS u
  ON u.id = oi.user_id
WHERE oi.status = 'Complete'
  AND o.status = 'Complete'
  AND p.brand = @brand
  AND DATE(oi.created_at) >= @start
  AND DATE(oi.created_at) < @end
GROUP BY month
ORDER BY month
LIMIT 20

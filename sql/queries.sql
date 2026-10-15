-- Q1. Overall order status counts
SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY order_count DESC;

-- Q2. Completed orders and revenue by month
SELECT
    substr(order_ts, 1, 7) AS order_month,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue_inr
FROM orders
WHERE status = 'completed'
GROUP BY substr(order_ts, 1, 7)
ORDER BY order_month;

-- Q3. Top 10 products by completed-order revenue
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(*) AS completed_order_count,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.product_id, p.product_name, p.category
ORDER BY revenue_inr DESC
LIMIT 10;

-- Q4. Customers with the highest completed-order spend
SELECT
    c.customer_id,
    c.city,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spend_inr
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY total_spend_inr DESC
LIMIT 10;

-- Q5. Cancellation rate by city
SELECT
    c.city,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        AS cancelled_orders,
    ROUND(
        100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    ) AS cancellation_rate_pct
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.city
ORDER BY cancellation_rate_pct DESC;

-- Q6. Average delivery time by city for completed orders
SELECT
    c.city,
    COUNT(o.delivery_minutes) AS orders_with_delivery_data,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
  AND o.delivery_minutes IS NOT NULL
GROUP BY c.city
ORDER BY avg_delivery_minutes DESC;

-- Q7. Revenue by product category
SELECT
    p.category,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr,
    ROUND(AVG(o.order_amount), 2) AS avg_order_amount_inr
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue_inr DESC;
